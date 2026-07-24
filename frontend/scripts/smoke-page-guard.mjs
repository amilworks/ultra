const safeMethods = new Set(["GET", "HEAD", "OPTIONS"]);
const safeReadPostPaths = new Set(["/v2/bisque/search"]);

export async function attachSmokePageGuard(page, { baseUrl, typographyAudit }) {
  const baseOrigin = new URL(baseUrl).origin;
  const blockedRequests = [];

  const onRequest = (request) => {
    typographyAudit.recordAttempt({
      resourceType: request.resourceType(),
      url: request.url(),
    });
  };
  const onResponse = (response) => {
    const request = response.request();
    typographyAudit.recordResponse({
      resourceType: request.resourceType(),
      url: request.url(),
      status: response.status(),
    });
  };
  const onRequestFailed = (request) => {
    typographyAudit.recordFailure({
      resourceType: request.resourceType(),
      url: request.url(),
      errorText: request.failure()?.errorText,
    });
  };
  const guardRoute = async (route) => {
    const request = route.request();
    const requestUrl = new URL(request.url());
    const methodIsSafe =
      safeMethods.has(request.method()) ||
      (request.method() === "POST" && safeReadPostPaths.has(requestUrl.pathname));
    if (
      !methodIsSafe ||
      requestUrl.origin !== baseOrigin
    ) {
      blockedRequests.push({
        method: request.method(),
        resourceType: request.resourceType(),
        url: request.url(),
      });
      await route.abort("blockedbyclient");
      return;
    }
    await route.fallback();
  };

  page.on("request", onRequest);
  page.on("response", onResponse);
  page.on("requestfailed", onRequestFailed);
  await page.route("**/*", guardRoute);

  return {
    blockedRequests,
    assertNoBlockedRequests(assert, caseName) {
      assert(
        blockedRequests.length === 0,
        `${caseName}: smoke blocked unsafe request(s): ${blockedRequests
          .map(({ method, url }) => `${method} ${url}`)
          .join(", ")}`
      );
    },
    async detach() {
      page.off("request", onRequest);
      page.off("response", onResponse);
      page.off("requestfailed", onRequestFailed);
      await page.unroute("**/*", guardRoute);
    },
  };
}
