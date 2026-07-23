const isTypographyRequest = ({ resourceType, url }) =>
  resourceType === "font" ||
  resourceType === "stylesheet" ||
  /\.(?:css|woff2?)(?:[?#]|$)/i.test(url);

export const createTypographyRequestAudit = (baseUrl) => {
  const baseOrigin = new URL(baseUrl).origin;
  const attempts = new Map();
  const failures = [];

  const recordAttempt = ({ resourceType, url }) => {
    if (!isTypographyRequest({ resourceType, url })) {
      return;
    }
    const attempt = attempts.get(url) ?? {
      type: resourceType,
      url,
      responseStatuses: [],
    };
    attempts.set(url, attempt);
  };

  const recordResponse = ({ resourceType, url, status }) => {
    if (!isTypographyRequest({ resourceType, url })) {
      return;
    }
    recordAttempt({ resourceType, url });
    attempts.get(url).responseStatuses.push(status);
  };

  const recordFailure = ({ resourceType, url, errorText }) => {
    if (!isTypographyRequest({ resourceType, url })) {
      return;
    }
    recordAttempt({ resourceType, url });
    failures.push({ type: resourceType, url, errorText });
  };

  const assertLocalSuccess = (assert, caseName) => {
    const attempted = [...attempts.values()];
    assert(attempted.length > 0, `${caseName}: no font/CSS requests were attempted`);
    for (const attempt of attempted) {
      assert(
        new URL(attempt.url).origin === baseOrigin,
        `${caseName}: attempted remote font/CSS request ${attempt.url}`
      );
    }
    assert(
      failures.length === 0,
      `${caseName}: font/CSS requestfailed ${failures
        .map(({ errorText, url }) => `${errorText || "unknown"} ${url}`)
        .join(", ")}`
    );
    const successfulLocal = attempted.filter(
      ({ url, responseStatuses }) =>
        new URL(url).origin === baseOrigin &&
        responseStatuses.some((status) => status >= 200 && status < 400)
    );
    assert(successfulLocal.length > 0, `${caseName}: no local font/CSS response succeeded`);
    return { attempted, successfulLocal };
  };

  return { recordAttempt, recordResponse, recordFailure, assertLocalSuccess };
};
