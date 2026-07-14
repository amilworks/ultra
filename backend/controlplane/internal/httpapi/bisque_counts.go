package httpapi

import (
	"context"
	"encoding/xml"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
)

// This file adds the reliable count/annotation primitives that the generic
// bisque_search path could not express, which is what produced the wrong
// answers in production (a dataset's image count collapsed to the whole
// owner,shared visibility pool, and "which images have annotations" was
// unanswerable so it was fabricated as 0). Three read-only capabilities:
//
//   - DatasetMembers       — the member count + enumeration (with names) of a
//                            dataset, read from its /value sub-collection (not a
//                            global image search).
//   - ImageAnnotationCount — the number of graphical (gobject) annotation SHAPES
//                            on one image, read from the image's /gobject
//                            sub-collection at view=deep, grouped by class label.
//   - ImagesWithAnnotations — answers "how many images in dataset X have
//                            annotations" by scanning members with bounded
//                            concurrency.
//
// Two BisQue traps this code exists to avoid, both verified live:
//  1. Counting via a scoped image search / count_all returns every image the
//     user can see, not a dataset's members.
//  2. Annotation SHAPES are nested. BisQue stores gobjects as a tree —
//     e.g. gt2 -> {burrow, prairie_dog, prairie_dog_in_burrow} -> many
//     <rectangle> shapes. The image's own view=deep TRUNCATES this (it returns
//     the class groups with zero children), and the /gobject?view=count
//     sub-collection is ACL-filtered and undercounts. Only the image's
//     /gobject?view=deep sub-collection returns the full tree with the actual
//     shape elements, so annotation counts must walk that tree and count the
//     primitive shapes (rectangle/polygon/point/...), not the group containers.
//
// All calls run under the caller's linked BisQue credentials (basic auth);
// shared datasets' member images are individually ACL-gated.

var bisqueUniqPattern = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

// bisqueShapeElementTags are BisQue's primitive annotation-shape elements — the
// actual marks a user draws. They are distinct from the <gobject> element, which
// is a container/label grouping shapes (e.g. a class name like "burrow").
var bisqueShapeElementTags = map[string]bool{
	"point": true, "rectangle": true, "polygon": true, "polyline": true,
	"circle": true, "ellipse": true, "line": true, "square": true,
	"freehand": true, "label": true,
}

const (
	bisqueAnnotationScanConcurrency = 16
	bisqueMaxAnnotationScan         = 8000
	bisqueMaxMemberEnumeration      = 100000
)

// --- request/response types ---

type bisqueDatasetMembersRequest struct {
	DatasetUniq string `json:"dataset_uniq"`
	ResourceURI string `json:"resource_uri"`
	Limit       int    `json:"limit"`
	Offset      int    `json:"offset"`
}

type bisqueDatasetMembersResponse struct {
	DatasetUniq string           `json:"dataset_uniq"`
	MemberCount int              `json:"member_count"`
	Offset      int              `json:"offset"`
	Members     []BisqueResource `json:"members"`
}

type bisqueImageAnnotationsRequest struct {
	ImageUniq   string `json:"image_uniq"`
	ResourceURI string `json:"resource_uri"`
}

type bisqueImageAnnotationsResponse struct {
	ImageUniq       string         `json:"image_uniq"`
	Name            string         `json:"name,omitempty"`
	AnnotationCount int            `json:"annotation_count"`
	GroupCount      int            `json:"group_count,omitempty"`
	LabelCounts     map[string]int `json:"label_counts,omitempty"`
	ClientViewURL   string         `json:"client_view_url,omitempty"`
}

type bisqueDatasetAnnotationsRequest struct {
	DatasetUniq string `json:"dataset_uniq"`
	ResourceURI string `json:"resource_uri"`
	MaxImages   int    `json:"max_images"`
}

type bisqueAnnotatedImage struct {
	ResourceUniq    string         `json:"resource_uniq"`
	Name            string         `json:"name,omitempty"`
	AnnotationCount int            `json:"annotation_count"`
	LabelCounts     map[string]int `json:"label_counts,omitempty"`
	ClientViewURL   string         `json:"client_view_url,omitempty"`
}

type bisqueDatasetAnnotationsResponse struct {
	DatasetUniq           string                 `json:"dataset_uniq"`
	MemberCount           int                    `json:"member_count"`
	ImagesChecked         int                    `json:"images_checked"`
	ImagesWithAnnotations int                    `json:"images_with_annotations"`
	TotalAnnotations      int                    `json:"total_annotations"`
	LabelTotals           map[string]int         `json:"label_totals,omitempty"`
	Inaccessible          int                    `json:"inaccessible"`
	Truncated             bool                   `json:"truncated"`
	AnnotatedImages       []bisqueAnnotatedImage `json:"annotated_images"`
}

// bisqueImageAnnotationInfo is the parsed gobject-tree summary for one image.
type bisqueImageAnnotationInfo struct {
	Count       int            // primitive annotation shapes (the actual marks)
	Groups      int            // gobject group/label containers
	Name        string         // image name (filled by the caller when known)
	LabelCounts map[string]int // shape counts by nearest named class label
}

func (info *bisqueImageAnnotationInfo) record(group, ownLabel string) {
	key := strings.TrimSpace(group)
	if key == "" {
		key = strings.TrimSpace(ownLabel)
	}
	if key == "" {
		key = "ungrouped"
	}
	if info.LabelCounts == nil {
		info.LabelCounts = map[string]int{}
	}
	info.LabelCounts[key]++
}

// --- view=count primitive ---

// parseBisqueCountTag extracts the <tag name="count" value="N"> child that
// BisQue's view=count operator returns. The count lives in a CHILD tag, not on
// the <resource> wrapper's attributes, which is why the old attribute-based
// bisqueSearchCount never found it and always fell back to paging.
func parseBisqueCountTag(data []byte) (int, bool) {
	var root bisqueXMLResource
	if err := xml.Unmarshal(data, &root); err != nil {
		return 0, false
	}
	return bisqueCountTagValue(root)
}

func bisqueCountTagValue(res bisqueXMLResource) (int, bool) {
	for _, tag := range res.Tags {
		if strings.EqualFold(strings.TrimSpace(tag.Name), "count") {
			if n, err := strconv.Atoi(strings.TrimSpace(tag.Value)); err == nil && n >= 0 {
				return n, true
			}
		}
	}
	for _, child := range res.Children {
		if n, ok := bisqueCountTagValue(child); ok {
			return n, true
		}
	}
	return 0, false
}

// countViaViewCount asks BisQue for an authoritative total using the view=count
// operator instead of paging the whole pool and summing. Returns (count, true)
// on success; (0, false) when the server does not answer with a count tag.
func (service *BisqueService) countViaViewCount(ctx context.Context, resourceType string, req bisqueSearchRequest, credentials BisqueCredentials) (int, bool, error) {
	searchURL, err := service.searchURL(resourceType, req, 1, 0, false)
	if err != nil {
		return 0, false, err
	}
	parsed, err := url.Parse(searchURL)
	if err != nil {
		return 0, false, err
	}
	query := parsed.Query()
	query.Set("view", "count")
	query.Del("limit")
	query.Del("offset")
	parsed.RawQuery = query.Encode()
	data, _, err := service.fetch(ctx, http.MethodGet, parsed.String(), nil, "", credentials)
	if err != nil {
		return 0, false, err
	}
	count, ok := parseBisqueCountTag(data)
	return count, ok, nil
}

// --- dataset members ---

func (service *BisqueService) DatasetMembers(ctx context.Context, datasetRef string, limit int, offset int, credentials BisqueCredentials) (bisqueDatasetMembersResponse, error) {
	if service == nil {
		return bisqueDatasetMembersResponse{}, bisqueClientError("BisQue integration is not configured")
	}
	uniq := bisqueResourceUniq(datasetRef)
	if !bisqueUniqPattern.MatchString(uniq) {
		return bisqueDatasetMembersResponse{}, bisqueClientError("a BisQue dataset resource_uniq is required")
	}
	base := "/data_service/dataset/" + url.PathEscape(uniq) + "/value"

	// Authoritative accessible member count via the /value sub-collection.
	total := 0
	if data, _, err := service.fetch(ctx, http.MethodGet, service.endpoint(base+"?view=count"), nil, "", credentials); err == nil {
		if n, ok := parseBisqueCountTag(data); ok {
			total = n
		}
	}

	if offset < 0 {
		offset = 0
	}
	if limit <= 0 {
		limit = 200
	}
	if limit > bisqueMaxMemberEnumeration {
		limit = bisqueMaxMemberEnumeration
	}
	membersURL := service.endpoint(base + "?view=short&limit=" + strconv.Itoa(limit) + "&offset=" + strconv.Itoa(offset))
	data, _, err := service.fetch(ctx, http.MethodGet, membersURL, nil, "", credentials)
	if err != nil {
		return bisqueDatasetMembersResponse{}, err
	}
	members, listed := parseBisqueSearchResponse(data)
	if total == 0 {
		total = listed
	}
	return bisqueDatasetMembersResponse{
		DatasetUniq: uniq,
		MemberCount: total,
		Offset:      offset,
		Members:     service.withLinks(members),
	}, nil
}

// --- image annotation (gobject) count ---

func (service *BisqueService) ImageAnnotationCount(ctx context.Context, imageRef string, credentials BisqueCredentials) (bisqueImageAnnotationInfo, error) {
	if service == nil {
		return bisqueImageAnnotationInfo{}, bisqueClientError("BisQue integration is not configured")
	}
	uniq := bisqueResourceUniq(imageRef)
	if !bisqueUniqPattern.MatchString(uniq) {
		return bisqueImageAnnotationInfo{}, bisqueClientError("a BisQue image resource_uniq is required")
	}
	// The /gobject sub-collection at view=deep is the only response that returns
	// the FULL nested gobject tree (the image's own view=deep truncates it).
	deepURL := service.endpoint("/data_service/" + url.PathEscape(uniq) + "/gobject?view=deep")
	data, _, err := service.fetch(ctx, http.MethodGet, deepURL, nil, "", credentials)
	if err != nil {
		return bisqueImageAnnotationInfo{}, err
	}
	return bisqueGobjectAnnotationStats(data), nil
}

// bisqueGobjectAnnotationStats walks a /gobject?view=deep tree and counts the
// primitive annotation SHAPES (rectangle/polygon/point/...) — the actual marks
// — grouped by their nearest named gobject class label (e.g. "burrow"). Plain
// <gobject> nodes are class containers, not marks, and are counted only in
// Groups. A <gobject> that is itself a drawn shape (carries vertices and has no
// child gobjects or shapes) is counted as a mark.
func bisqueGobjectAnnotationStats(data []byte) bisqueImageAnnotationInfo {
	info := bisqueImageAnnotationInfo{LabelCounts: map[string]int{}}
	var root bisqueXMLResource
	if err := xml.Unmarshal(data, &root); err != nil {
		return info
	}
	bisqueWalkAnnotations(root, "", &info)
	return info
}

func bisqueWalkAnnotations(node bisqueXMLResource, group string, info *bisqueImageAnnotationInfo) {
	for i := range node.Children {
		child := node.Children[i]
		tag := strings.ToLower(strings.TrimSpace(child.XMLName.Local))
		switch {
		case tag == "vertex" || tag == "tag" || tag == "value" || tag == "template":
			continue
		case tag == "gobject":
			if bisqueNodeHasVertex(child) && !bisqueNodeHasChildTag(child, "gobject") && !bisqueNodeHasShapeChild(child) {
				info.Count++
				info.record(group, firstNonEmpty(strings.TrimSpace(child.Name), strings.TrimSpace(child.Type)))
				continue
			}
			info.Groups++
			next := firstNonEmpty(strings.TrimSpace(child.Name), strings.TrimSpace(child.Type))
			if next == "" {
				next = group
			}
			bisqueWalkAnnotations(child, next, info)
		case bisqueShapeElementTags[tag]:
			info.Count++
			info.record(group, "")
		default:
			// Unknown wrapper element (e.g. the <resource> sub-collection root or
			// an <image>): descend without changing the class group.
			bisqueWalkAnnotations(child, group, info)
		}
	}
}

func bisqueNodeHasVertex(node bisqueXMLResource) bool {
	return bisqueNodeHasChildTag(node, "vertex")
}

func bisqueNodeHasChildTag(node bisqueXMLResource, tag string) bool {
	for _, child := range node.Children {
		if strings.EqualFold(child.XMLName.Local, tag) {
			return true
		}
	}
	return false
}

func bisqueNodeHasShapeChild(node bisqueXMLResource) bool {
	for _, child := range node.Children {
		if bisqueShapeElementTags[strings.ToLower(strings.TrimSpace(child.XMLName.Local))] {
			return true
		}
	}
	return false
}

// --- dataset-wide "images with annotations" ---

func (service *BisqueService) ImagesWithAnnotations(ctx context.Context, datasetRef string, maxImages int, credentials BisqueCredentials) (bisqueDatasetAnnotationsResponse, error) {
	if service == nil {
		return bisqueDatasetAnnotationsResponse{}, bisqueClientError("BisQue integration is not configured")
	}
	members, err := service.DatasetMembers(ctx, datasetRef, bisqueMaxMemberEnumeration, 0, credentials)
	if err != nil {
		return bisqueDatasetAnnotationsResponse{}, err
	}

	limit := maxImages
	if limit <= 0 || limit > bisqueMaxAnnotationScan {
		limit = bisqueMaxAnnotationScan
	}
	toScan := members.Members
	truncated := false
	if len(toScan) > limit {
		toScan = toScan[:limit]
		truncated = true
	}

	type probe struct {
		info         bisqueImageAnnotationInfo
		member       BisqueResource
		inaccessible bool
	}
	probes := make([]probe, len(toScan))
	semaphore := make(chan struct{}, bisqueAnnotationScanConcurrency)
	var wg sync.WaitGroup
	for i := range toScan {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()
			member := toScan[idx]
			info, probeErr := service.ImageAnnotationCount(ctx, member.ResourceUniq, credentials)
			if probeErr != nil {
				probes[idx] = probe{member: member, inaccessible: true}
				return
			}
			probes[idx] = probe{info: info, member: member}
		}(i)
	}
	wg.Wait()

	response := bisqueDatasetAnnotationsResponse{
		DatasetUniq:     members.DatasetUniq,
		MemberCount:     members.MemberCount,
		Truncated:       truncated,
		LabelTotals:     map[string]int{},
		AnnotatedImages: []bisqueAnnotatedImage{},
	}
	for _, p := range probes {
		if p.inaccessible {
			response.Inaccessible++
			continue
		}
		response.ImagesChecked++
		if p.info.Count <= 0 {
			continue
		}
		response.ImagesWithAnnotations++
		response.TotalAnnotations += p.info.Count
		for label, n := range p.info.LabelCounts {
			response.LabelTotals[label] += n
		}
		response.AnnotatedImages = append(response.AnnotatedImages, bisqueAnnotatedImage{
			ResourceUniq:    p.member.ResourceUniq,
			Name:            p.member.Name,
			AnnotationCount: p.info.Count,
			LabelCounts:     p.info.LabelCounts,
			ClientViewURL:   p.member.ClientViewURL,
		})
	}
	if len(response.LabelTotals) == 0 {
		response.LabelTotals = nil
	}
	return response, nil
}

// bisqueResourceUniq extracts a bare resource_uniq from either a bare uniq or a
// full data_service URI (e.g. https://host/data_service/00-abc -> 00-abc).
func bisqueResourceUniq(ref string) string {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return ""
	}
	if strings.Contains(ref, "/") {
		return pathBaseFromURL(ref)
	}
	return ref
}

// --- HTTP handlers ---

func (deps ServerDeps) handleBisqueDatasetMembers(w http.ResponseWriter, r *http.Request) {
	if deps.Bisque == nil {
		writeBisqueNotConfigured(w)
		return
	}
	authority, authorized := deps.authorizeBisqueRequest(w, r)
	if !authorized {
		return
	}
	var req bisqueDatasetMembersRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	ref := firstNonEmpty(strings.TrimSpace(req.DatasetUniq), strings.TrimSpace(req.ResourceURI))
	response, err := deps.Bisque.DatasetMembers(r.Context(), ref, req.Limit, req.Offset, authority.Credentials)
	if err != nil {
		writeBisqueError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

func (deps ServerDeps) handleBisqueImageAnnotations(w http.ResponseWriter, r *http.Request) {
	if deps.Bisque == nil {
		writeBisqueNotConfigured(w)
		return
	}
	authority, authorized := deps.authorizeBisqueRequest(w, r)
	if !authorized {
		return
	}
	var req bisqueImageAnnotationsRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	ref := firstNonEmpty(strings.TrimSpace(req.ImageUniq), strings.TrimSpace(req.ResourceURI))
	info, err := deps.Bisque.ImageAnnotationCount(r.Context(), ref, authority.Credentials)
	if err != nil {
		writeBisqueError(w, err)
		return
	}
	uniq := bisqueResourceUniq(ref)
	name := info.Name
	if name == "" {
		// Best-effort image name for the single-image response; the /gobject tree
		// itself carries no image name.
		if resource, ferr := bisqueFetchImageName(r.Context(), deps.Bisque, uniq, authority.Credentials); ferr == nil {
			name = resource
		}
	}
	writeJSON(w, http.StatusOK, bisqueImageAnnotationsResponse{
		ImageUniq:       uniq,
		Name:            name,
		AnnotationCount: info.Count,
		GroupCount:      info.Groups,
		LabelCounts:     info.LabelCounts,
		ClientViewURL:   deps.Bisque.clientViewURL(deps.Bisque.endpoint("/data_service/image/" + uniq)),
	})
}

func bisqueFetchImageName(ctx context.Context, service *BisqueService, uniq string, credentials BisqueCredentials) (string, error) {
	resource, err := service.fetchResource(ctx, service.endpoint("/data_service/image/"+url.PathEscape(uniq)+"?view=short"), credentials)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(resource.Name), nil
}

func (deps ServerDeps) handleBisqueDatasetAnnotations(w http.ResponseWriter, r *http.Request) {
	if deps.Bisque == nil {
		writeBisqueNotConfigured(w)
		return
	}
	authority, authorized := deps.authorizeBisqueRequest(w, r)
	if !authorized {
		return
	}
	var req bisqueDatasetAnnotationsRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	ref := firstNonEmpty(strings.TrimSpace(req.DatasetUniq), strings.TrimSpace(req.ResourceURI))
	response, err := deps.Bisque.ImagesWithAnnotations(r.Context(), ref, req.MaxImages, authority.Credentials)
	if err != nil {
		writeBisqueError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}
