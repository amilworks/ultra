package httpapi

import (
	"bytes"
	"context"
	"encoding/xml"
	"fmt"
	"html"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// BisQue analysis modules run on BisQue's own engine. Ultra lists and describes
// them, and launches them by passing BisQue resource URIs as inputs, so neither
// the data nor the analysis is copied into Ultra.

const (
	bisqueModuleCatalogTTL   = 10 * time.Minute
	bisqueModuleHelpMaxRunes = 4000
)

var (
	bisqueModuleCatalogCache sync.Map // rootURL + "\x00" + username -> bisqueModuleCatalogEntry
	bisqueHTMLScriptPattern  = regexp.MustCompile(`(?is)<(?:script|style)[^>]*>.*?</(?:script|style)>`)
	bisqueHTMLTagPattern     = regexp.MustCompile(`(?s)<[^>]+>`)
)

type bisqueModuleCatalogEntry struct {
	modules   []BisqueModule
	fetchedAt time.Time
}

type BisqueModuleParam struct {
	Name     string            `json:"name"`
	Type     string            `json:"type,omitempty"`
	Default  string            `json:"default,omitempty"`
	System   bool              `json:"system,omitempty"`
	Template map[string]string `json:"template,omitempty"`
}

type BisqueModule struct {
	Name         string              `json:"name"`
	Title        string              `json:"title,omitempty"`
	Description  string              `json:"description,omitempty"`
	Authors      string              `json:"authors,omitempty"`
	ResourceURI  string              `json:"resource_uri"`
	ResourceUniq string              `json:"resource_uniq,omitempty"`
	UpdatedAt    string              `json:"updated_at,omitempty"`
	Inputs       []BisqueModuleParam `json:"inputs,omitempty"`
	Outputs      []BisqueModuleParam `json:"outputs,omitempty"`
	HelpURL      string              `json:"help_url,omitempty"`
	ModuleURL    string              `json:"module_url,omitempty"`
	Help         string              `json:"help,omitempty"`
	ExampleRun   *BisqueResource     `json:"example_run,omitempty"`
}

type bisqueModuleSummary struct {
	Name        string              `json:"name"`
	Title       string              `json:"title,omitempty"`
	Description string              `json:"description,omitempty"`
	Inputs      []BisqueModuleParam `json:"inputs,omitempty"`
	Outputs     []string            `json:"outputs,omitempty"`
	MatchScore  int                 `json:"match_score,omitempty"`
}

type bisqueModuleListRequest struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

type bisqueModuleDescribeRequest struct {
	Name       string `json:"name"`
	ModuleName string `json:"module_name"`
}

type bisqueModuleExecuteRequest struct {
	ModuleName string            `json:"module_name"`
	Inputs     map[string]string `json:"inputs"`
}

type bisqueModuleExecuteMutationRequest struct {
	ModuleName string            `json:"module_name"`
	Inputs     map[string]string `json:"inputs"`
}

type bisqueModuleExecuteResponse struct {
	OK            bool           `json:"ok"`
	ModuleName    string         `json:"module_name"`
	Status        string         `json:"status,omitempty"`
	MexURI        string         `json:"mex_uri,omitempty"`
	MexUniq       string         `json:"mex_uniq,omitempty"`
	ClientViewURL string         `json:"client_view_url,omitempty"`
	Run           BisqueResource `json:"run"`
}

type bisqueModuleExecution struct {
	Module BisqueModule
	Inputs map[string]string
	MexXML []byte
}

// bisqueXMLNode is a generic element used to read module definitions (whose
// shape varies by module) and to build MEX documents.
type bisqueXMLNode struct {
	XMLName  xml.Name
	Attrs    []xml.Attr      `xml:",any,attr"`
	Text     string          `xml:",chardata"`
	Children []bisqueXMLNode `xml:",any"`
}

func (node bisqueXMLNode) attr(name string) string {
	for _, attr := range node.Attrs {
		if attr.Name.Local == name {
			return attr.Value
		}
	}
	return ""
}

func (node *bisqueXMLNode) setAttr(name string, value string) {
	for i := range node.Attrs {
		if node.Attrs[i].Name.Local == name {
			node.Attrs[i].Value = value
			return
		}
	}
	node.Attrs = append(node.Attrs, xml.Attr{Name: xml.Name{Local: name}, Value: value})
}

func (node bisqueXMLNode) isTag() bool {
	return strings.EqualFold(node.XMLName.Local, "tag")
}

func collectBisqueModuleNodes(node bisqueXMLNode, out *[]bisqueXMLNode) {
	if strings.EqualFold(node.XMLName.Local, "module") {
		*out = append(*out, node)
		return
	}
	for _, child := range node.Children {
		collectBisqueModuleNodes(child, out)
	}
}

func bisqueTagText(node bisqueXMLNode) string {
	return firstNonEmpty(strings.TrimSpace(node.attr("value")), strings.TrimSpace(node.Text))
}

func bisqueModuleFromNode(node bisqueXMLNode) BisqueModule {
	module := BisqueModule{
		Name:         strings.TrimSpace(node.attr("name")),
		ResourceURI:  strings.TrimSpace(node.attr("uri")),
		ResourceUniq: strings.TrimSpace(node.attr("resource_uniq")),
		UpdatedAt:    firstNonEmpty(strings.TrimSpace(node.attr("ts")), strings.TrimSpace(node.attr("created"))),
	}
	for _, child := range node.Children {
		if !child.isTag() {
			continue
		}
		switch strings.ToLower(strings.TrimSpace(child.attr("name"))) {
		case "title":
			module.Title = bisqueTagText(child)
		case "description":
			module.Description = bisqueTagText(child)
		case "authors":
			module.Authors = bisqueTagText(child)
		case "inputs":
			module.Inputs = bisqueModuleParams(child)
		case "outputs":
			module.Outputs = bisqueModuleParams(child)
		}
	}
	return module
}

func bisqueModuleParams(container bisqueXMLNode) []BisqueModuleParam {
	params := []BisqueModuleParam{}
	for _, child := range container.Children {
		if !child.isTag() {
			continue
		}
		name := strings.TrimSpace(child.attr("name"))
		if name == "" {
			continue
		}
		paramType := strings.TrimSpace(child.attr("type"))
		param := BisqueModuleParam{
			Name:    name,
			Type:    paramType,
			Default: strings.TrimSpace(child.attr("value")),
			System:  bisqueModuleParamIsSystem(name, paramType),
		}
		for _, sub := range child.Children {
			// Module XML writes templates as a <template> element; older modules
			// use <tag name="template">. Keys such as accepted_type may repeat.
			isTemplate := strings.EqualFold(sub.XMLName.Local, "template") ||
				(sub.isTag() && strings.EqualFold(strings.TrimSpace(sub.attr("name")), "template"))
			if !isTemplate {
				continue
			}
			template := map[string]string{}
			for _, entry := range sub.Children {
				key := strings.TrimSpace(entry.attr("name"))
				value := bisqueTagText(entry)
				if key == "" || value == "" {
					continue
				}
				if existing, ok := template[key]; ok {
					value = existing + ", " + value
				}
				template[key] = truncateRunes(value, 200)
			}
			if len(template) > 0 {
				param.Template = template
			}
		}
		params = append(params, param)
	}
	return params
}

func bisqueModuleParamIsSystem(name string, paramType string) bool {
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(paramType)), "system") {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "mex_url", "bisque_token":
		return true
	}
	return false
}

// withoutSystemMexParams drops BisQue-filled system inputs. bisque_token is a
// live credential for an in-flight run and must never reach a model.
func withoutSystemMexParams(params []BisqueMexParam) []BisqueMexParam {
	kept := make([]BisqueMexParam, 0, len(params))
	for _, param := range params {
		if bisqueModuleParamIsSystem(param.Name, param.Type) {
			continue
		}
		kept = append(kept, param)
	}
	return kept
}

// bisqueModuleParamRequired follows the module template the way BisQue's own
// run form does: an explicit allow_blank/allowBlank wins; otherwise resource
// inputs are required and scalar inputs are optional.
func bisqueModuleParamRequired(param BisqueModuleParam) bool {
	for _, key := range []string{"allow_blank", "allowBlank"} {
		if value, ok := param.Template[key]; ok {
			return !strings.HasPrefix(strings.ToLower(strings.TrimSpace(value)), "true")
		}
	}
	return bisqueModuleInputIsReference(param.Type)
}

func bisqueModuleInputIsReference(formalType string) bool {
	switch strings.ToLower(strings.TrimSpace(formalType)) {
	case "resource", "image", "file", "table", "dataset", "mex", "gobject":
		return true
	}
	return false
}

func truncateRunes(text string, limit int) string {
	runes := []rune(text)
	if len(runes) <= limit {
		return text
	}
	return string(runes[:limit]) + "…"
}

func bisqueHelpText(data []byte) string {
	text := bisqueHTMLScriptPattern.ReplaceAllString(string(data), " ")
	text = bisqueHTMLTagPattern.ReplaceAllString(text, " ")
	text = html.UnescapeString(text)
	return truncateRunes(strings.Join(strings.Fields(text), " "), bisqueModuleHelpMaxRunes)
}

// ListModules returns the modules currently registered with module_service (the
// runnable set), enriched with their data_service definitions. data_service
// also keeps retired registrations, so the module_service index is authoritative.
func (service *BisqueService) ListModules(ctx context.Context, credentials BisqueCredentials) ([]BisqueModule, error) {
	if service == nil {
		return nil, bisqueClientError("BisQue integration is not configured")
	}
	username, _ := service.credentialsForRequest(credentials)
	key := service.rootURL + "\x00" + username
	if cached, ok := bisqueModuleCatalogCache.Load(key); ok {
		entry := cached.(bisqueModuleCatalogEntry)
		if time.Since(entry.fetchedAt) < bisqueModuleCatalogTTL {
			return append([]BisqueModule(nil), entry.modules...), nil
		}
	}
	indexData, _, err := service.fetchNoRedirect(ctx, http.MethodGet, service.endpoint("/module_service/"), nil, "", credentials)
	if err != nil {
		return nil, err
	}
	var index bisqueXMLNode
	if err := xml.Unmarshal(indexData, &index); err != nil {
		return nil, bisqueServerError("BisQue module_service returned an unreadable module list")
	}
	active := []bisqueXMLNode{}
	collectBisqueModuleNodes(index, &active)

	definitions := map[string]bisqueXMLNode{}
	deepURL := service.endpoint("/data_service/module") + "?view=deep&limit=1000"
	if deepData, _, err := service.fetchNoRedirect(ctx, http.MethodGet, deepURL, nil, "", credentials); err == nil {
		var deep bisqueXMLNode
		if xml.Unmarshal(deepData, &deep) == nil {
			nodes := []bisqueXMLNode{}
			collectBisqueModuleNodes(deep, &nodes)
			for _, node := range nodes {
				if uri := strings.TrimSpace(node.attr("uri")); uri != "" {
					definitions[uri] = node
				}
			}
		}
	}

	modules := make([]BisqueModule, 0, len(active))
	for _, entry := range active {
		registered := bisqueModuleFromNode(entry)
		definition, found := definitions[registered.ResourceURI]
		if !found && registered.ResourceURI != "" {
			if fetched, err := service.fetchModuleDefinition(ctx, registered.ResourceURI, credentials); err == nil {
				definition, found = fetched, true
			}
		}
		module := registered
		if found {
			module = bisqueModuleFromNode(definition)
			module.Name = firstNonEmpty(registered.Name, module.Name)
			module.ResourceURI = firstNonEmpty(registered.ResourceURI, module.ResourceURI)
			module.ResourceUniq = firstNonEmpty(registered.ResourceUniq, module.ResourceUniq)
		}
		if module.Name == "" {
			continue
		}
		escaped := url.PathEscape(module.Name)
		module.HelpURL = service.endpoint("/module_service/" + escaped + "/help")
		module.ModuleURL = service.endpoint("/module_service/" + escaped + "/")
		modules = append(modules, module)
	}
	sort.Slice(modules, func(i, j int) bool {
		return strings.ToLower(modules[i].Name) < strings.ToLower(modules[j].Name)
	})
	bisqueModuleCatalogCache.Store(key, bisqueModuleCatalogEntry{modules: modules, fetchedAt: time.Now()})
	return append([]BisqueModule(nil), modules...), nil
}

func (service *BisqueService) fetchModuleDefinition(ctx context.Context, moduleURI string, credentials BisqueCredentials) (bisqueXMLNode, error) {
	deepURL, err := service.deepViewURL(moduleURI)
	if err != nil {
		return bisqueXMLNode{}, err
	}
	data, _, err := service.fetchNoRedirect(ctx, http.MethodGet, deepURL, nil, "", credentials)
	if err != nil {
		return bisqueXMLNode{}, err
	}
	var root bisqueXMLNode
	if err := xml.Unmarshal(data, &root); err != nil {
		return bisqueXMLNode{}, bisqueServerError("BisQue module definition is unreadable")
	}
	nodes := []bisqueXMLNode{}
	collectBisqueModuleNodes(root, &nodes)
	if len(nodes) == 0 {
		return bisqueXMLNode{}, bisqueServerError("BisQue resource is not a module definition")
	}
	return nodes[0], nil
}

func (service *BisqueService) findModule(ctx context.Context, name string, credentials BisqueCredentials) (BisqueModule, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return BisqueModule{}, bisqueClientError("a BisQue module name is required")
	}
	modules, err := service.ListModules(ctx, credentials)
	if err != nil {
		return BisqueModule{}, err
	}
	for _, module := range modules {
		if module.Name == name {
			return module, nil
		}
	}
	for _, module := range modules {
		if strings.EqualFold(module.Name, name) {
			return module, nil
		}
	}
	return BisqueModule{}, bisqueClientError(fmt.Sprintf(
		"BisQue module %q is not registered on this server; list the modules to see available names", name))
}

// DescribeModule adds the help page and the inputs of the most recent finished
// run, which shows concrete values (e.g. a weights file already on BisQue).
func (service *BisqueService) DescribeModule(ctx context.Context, name string, credentials BisqueCredentials) (BisqueModule, error) {
	module, err := service.findModule(ctx, name, credentials)
	if err != nil {
		return BisqueModule{}, err
	}
	if data, _, err := service.fetchNoRedirect(ctx, http.MethodGet, module.HelpURL, nil, "", credentials); err == nil {
		module.Help = bisqueHelpText(data)
	}
	if run, ok := service.latestFinishedModuleRun(ctx, module.Name, credentials); ok {
		module.ExampleRun = &run
	}
	return module, nil
}

func (service *BisqueService) latestFinishedModuleRun(ctx context.Context, name string, credentials BisqueCredentials) (BisqueResource, bool) {
	query := url.Values{}
	query.Set("tag_query", "@name:"+name)
	query.Set("tag_order", "@ts:desc")
	query.Set("view", "short")
	query.Set("limit", "25")
	query.Set("wpublic", "1")
	searchURL := service.endpoint("/data_service/mex") + "?" + query.Encode()
	data, _, err := service.fetchNoRedirect(ctx, http.MethodGet, searchURL, nil, "", credentials)
	if err != nil {
		return BisqueResource{}, false
	}
	for _, candidate := range parseBisqueResources(data) {
		if !strings.EqualFold(candidate.ResourceType, "mex") ||
			!strings.EqualFold(candidate.Status, "FINISHED") ||
			!strings.EqualFold(candidate.ModuleName, name) {
			continue
		}
		run, err := service.GetModuleRun(ctx, candidate.ResourceURI, credentials)
		if err != nil {
			continue
		}
		return run, true
	}
	return BisqueResource{}, false
}

// PrepareModuleExecution validates a run request against the module's formal
// inputs and builds the MEX document. Every resource input must be a reference
// on the configured BisQue root, which is what keeps execution by-reference.
func (service *BisqueService) PrepareModuleExecution(ctx context.Context, req bisqueModuleExecuteRequest, credentials BisqueCredentials) (bisqueModuleExecution, error) {
	module, err := service.findModule(ctx, req.ModuleName, credentials)
	if err != nil {
		return bisqueModuleExecution{}, err
	}
	if len(module.Inputs) == 0 {
		return bisqueModuleExecution{}, bisqueClientError(fmt.Sprintf("BisQue module %q has no input definition", module.Name))
	}

	formal := map[string]BisqueModuleParam{}
	userFacing := []string{}
	for _, input := range module.Inputs {
		formal[strings.ToLower(input.Name)] = input
		if !input.System {
			userFacing = append(userFacing, input.Name)
		}
	}
	provided := map[string]string{}
	unknown := []string{}
	for rawName, rawValue := range req.Inputs {
		input, ok := formal[strings.ToLower(strings.TrimSpace(rawName))]
		if !ok || input.System {
			unknown = append(unknown, rawName)
			continue
		}
		provided[input.Name] = strings.TrimSpace(rawValue)
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		return bisqueModuleExecution{}, bisqueClientError(fmt.Sprintf(
			"unknown inputs for %s: %s (valid inputs: %s)",
			module.Name, strings.Join(unknown, ", "), strings.Join(userFacing, ", ")))
	}

	// BisQue stores a run's inputs as plain name/type/value tags, with generic
	// "resource" inputs narrowed to the concrete type (image, file, ...) and
	// system inputs left for the engine to fill. Build exactly that shape.
	inputsNode := bisqueXMLNode{XMLName: xml.Name{Local: "tag"}}
	inputsNode.setAttr("name", "inputs")
	canonical := map[string]string{}
	missing := []string{}
	for _, input := range module.Inputs {
		node := bisqueXMLNode{XMLName: xml.Name{Local: "tag"}}
		node.setAttr("name", input.Name)
		paramType := input.Type
		if input.System {
			if paramType != "" {
				node.setAttr("type", paramType)
			}
			inputsNode.Children = append(inputsNode.Children, node)
			continue
		}
		value := provided[input.Name]
		if value == "" {
			value = input.Default
		}
		if value == "" {
			if bisqueModuleParamRequired(input) {
				missing = append(missing, input.Name)
				continue
			}
			if paramType != "" {
				node.setAttr("type", paramType)
			}
			inputsNode.Children = append(inputsNode.Children, node)
			continue
		}
		if bisqueModuleInputIsReference(paramType) || strings.Contains(value, "/data_service/") {
			if !strings.Contains(value, "/") {
				value = service.endpoint("/data_service/" + url.PathEscape(value))
			}
			normalized, err := service.normalizeAllowedURL(value)
			if err != nil {
				return bisqueModuleExecution{}, bisqueClientError(fmt.Sprintf(
					"input %q must reference a resource on the configured BisQue server", input.Name))
			}
			resource, err := service.fetchResource(ctx, normalized, credentials)
			if err != nil {
				return bisqueModuleExecution{}, bisqueClientError(fmt.Sprintf(
					"input %q could not be read on BisQue (check that it exists and is shared with you): %v", input.Name, err))
			}
			value = normalized
			if paramType == "" || strings.EqualFold(paramType, "resource") {
				if resourceType := strings.ToLower(strings.TrimSpace(resource.ResourceType)); resourceType != "" {
					paramType = resourceType
				}
			}
		}
		if paramType != "" {
			node.setAttr("type", paramType)
		}
		node.setAttr("value", value)
		canonical[input.Name] = value
		inputsNode.Children = append(inputsNode.Children, node)
	}
	if len(missing) > 0 {
		return bisqueModuleExecution{}, bisqueClientError(fmt.Sprintf(
			"missing required inputs for %s: %s", module.Name, strings.Join(missing, ", ")))
	}

	mex := bisqueXMLNode{XMLName: xml.Name{Local: "mex"}, Children: []bisqueXMLNode{inputsNode}}
	payload, err := xml.Marshal(mex)
	if err != nil {
		return bisqueModuleExecution{}, err
	}
	return bisqueModuleExecution{Module: module, Inputs: canonical, MexXML: payload}, nil
}

// SubmitModuleExecution posts the MEX to module_service, the same call BisQue's
// web client makes. BisQue schedules the run on its engine and returns at once.
func (service *BisqueService) SubmitModuleExecution(ctx context.Context, execution bisqueModuleExecution, credentials BisqueCredentials) (BisqueResource, error) {
	executeURL := service.endpoint("/module_service/" + url.PathEscape(execution.Module.Name) + "/execute")
	data, _, err := service.fetchNoRedirect(ctx, http.MethodPost, executeURL, bytes.NewReader(execution.MexXML), "text/xml", credentials)
	if err != nil {
		return BisqueResource{}, err
	}
	mex := bisqueFirstMexResource(service.withLinks(parseBisqueResources(data)))
	if mex == nil {
		return BisqueResource{}, bisqueServerError("BisQue did not return a module run for the execute request")
	}
	run := *mex
	if uniq := strings.TrimSpace(run.ResourceUniq); uniq != "" {
		// module_service echoes the run as /module_service/mex/<uniq>; the
		// data_service URI is the canonical one the viewer and run reader use.
		run.ResourceURI = service.endpoint("/data_service/" + url.PathEscape(uniq))
		run.ClientViewURL = service.clientViewURL(run.ResourceURI)
	}
	run.Inputs = withoutSystemMexParams(run.Inputs)
	return run, nil
}

func rankBisqueModules(modules []BisqueModule, query string) []bisqueModuleSummary {
	tokens := []string{}
	for _, word := range strings.FieldsFunc(strings.ToLower(query), func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9')
	}) {
		if len(word) < 3 {
			continue
		}
		if len(word) > 4 {
			word = strings.TrimSuffix(word, "s")
		}
		tokens = append(tokens, word)
	}
	summaries := make([]bisqueModuleSummary, 0, len(modules))
	for _, module := range modules {
		summary := bisqueModuleSummary{Name: module.Name, Title: module.Title, Description: module.Description}
		haystack := []string{strings.ToLower(module.Name), strings.ToLower(module.Title), strings.ToLower(module.Description)}
		for _, input := range module.Inputs {
			if input.System {
				continue
			}
			summary.Inputs = append(summary.Inputs, input)
			haystack = append(haystack, strings.ToLower(input.Name))
		}
		for _, output := range module.Outputs {
			summary.Outputs = append(summary.Outputs, output.Name)
		}
		text := strings.Join(haystack, " ")
		for _, token := range tokens {
			if strings.Contains(text, token) {
				summary.MatchScore++
			}
		}
		summaries = append(summaries, summary)
	}
	if len(tokens) > 0 {
		sort.SliceStable(summaries, func(i, j int) bool { return summaries[i].MatchScore > summaries[j].MatchScore })
	}
	return summaries
}

func (deps ServerDeps) handleBisqueModules(w http.ResponseWriter, r *http.Request) {
	if deps.Bisque == nil {
		writeBisqueNotConfigured(w)
		return
	}
	authority, authorized := deps.authorizeBisqueRequest(w, r)
	if !authorized {
		return
	}
	var req bisqueModuleListRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	modules, err := deps.Bisque.ListModules(r.Context(), authority.Credentials)
	if err != nil {
		writeBisqueError(w, err)
		return
	}
	summaries := rankBisqueModules(modules, req.Query)
	if req.Limit > 0 && req.Limit < len(summaries) {
		summaries = summaries[:req.Limit]
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":      true,
		"query":   strings.TrimSpace(req.Query),
		"count":   len(summaries),
		"total":   len(modules),
		"modules": summaries,
	})
}

func (deps ServerDeps) handleBisqueModuleDescribe(w http.ResponseWriter, r *http.Request) {
	if deps.Bisque == nil {
		writeBisqueNotConfigured(w)
		return
	}
	authority, authorized := deps.authorizeBisqueRequest(w, r)
	if !authorized {
		return
	}
	var req bisqueModuleDescribeRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	module, err := deps.Bisque.DescribeModule(r.Context(), firstNonEmpty(req.Name, req.ModuleName), authority.Credentials)
	if err != nil {
		writeBisqueError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "module": module})
}

func (deps ServerDeps) handleBisqueModuleExecute(w http.ResponseWriter, r *http.Request) {
	if deps.Bisque == nil {
		writeBisqueNotConfigured(w)
		return
	}
	authority, authorized := deps.authorizeBisqueRequest(w, r, domain.RemoteMutationIntentBisqueRunModule)
	if !authorized {
		return
	}
	var req bisqueModuleExecuteRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	// Unlike dataset creation, inputs are not limited to resources this run
	// uploaded or selected: analysing the user's existing BisQue data in place is
	// the purpose. BisQue enforces read access with the user's own credentials,
	// and every reference is pinned to the configured BisQue root.
	execution, err := deps.Bisque.PrepareModuleExecution(r.Context(), req, authority.Credentials)
	if err != nil {
		writeBisqueError(w, err)
		return
	}
	receipt, cached, replayed, err := deps.beginWorkerBisqueMutation(
		r.Context(), authority, "bisque.run_module",
		bisqueModuleExecuteMutationRequest{ModuleName: execution.Module.Name, Inputs: execution.Inputs},
	)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if replayed {
		writeJSON(w, http.StatusOK, cached)
		return
	}
	run, err := deps.Bisque.SubmitModuleExecution(r.Context(), execution, authority.Credentials)
	if err != nil {
		deps.markWorkerBisqueMutationAmbiguous(r.Context(), receipt, err)
		writeBisqueError(w, err)
		return
	}
	response := bisqueModuleExecuteResponse{
		OK:            true,
		ModuleName:    execution.Module.Name,
		Status:        run.Status,
		MexURI:        run.ResourceURI,
		MexUniq:       run.ResourceUniq,
		ClientViewURL: run.ClientViewURL,
		Run:           run,
	}
	if err := deps.completeWorkerBisqueMutation(r.Context(), receipt, response); err != nil {
		deps.markWorkerBisqueMutationAmbiguous(r.Context(), receipt, err)
		writeError(w, http.StatusServiceUnavailable, fmt.Errorf("BisQue module run started upstream but its durable receipt could not be sealed"))
		return
	}
	writeJSON(w, http.StatusOK, response)
}
