export const stopProcess = (child, termGraceMs = 2_000) =>
  new Promise((resolve) => {
    if (!child || child.exitCode !== null || child.signalCode !== null) {
      resolve();
      return;
    }

    let escalationTimer;
    const finish = () => {
      clearTimeout(escalationTimer);
      resolve();
    };
    child.once("exit", finish);
    child.kill("SIGTERM");
    escalationTimer = setTimeout(() => {
      if (child.exitCode === null && child.signalCode === null) {
        child.kill("SIGKILL");
      }
    }, termGraceMs);
  });

export const waitForProcess = (child, { label, timeoutMs, termGraceMs = 2_000 }) =>
  new Promise((resolve, reject) => {
    let settled = false;
    let timingOut = false;
    const finish = (error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeout);
      child.off("error", onError);
      child.off("exit", onExit);
      if (error) {
        reject(error);
      } else {
        resolve();
      }
    };
    const onError = (error) => finish(new Error(`${label} failed to start: ${error.message}`));
    const onExit = (code, signal) => {
      if (timingOut) {
        return;
      }
      if (code === 0) {
        finish();
        return;
      }
      finish(new Error(`${label} exited with status ${code ?? signal}`));
    };
    const timeout = setTimeout(async () => {
      if (settled) {
        return;
      }
      timingOut = true;
      await stopProcess(child, termGraceMs);
      finish(new Error(`${label} exceeded ${timeoutMs}ms`));
    }, timeoutMs);
    child.once("error", onError);
    child.once("exit", onExit);
    if (child.exitCode !== null || child.signalCode !== null) {
      onExit(child.exitCode, child.signalCode);
    }
  });

const waitForUnexpectedExit = (child, label) =>
  new Promise((_, reject) => {
    const onError = (error) => reject(new Error(`${label} failed: ${error.message}`));
    const onExit = (code, signal) =>
      reject(new Error(`${label} exited unexpectedly with status ${code ?? signal}`));
    child.once("error", onError);
    child.once("exit", onExit);
    if (child.exitCode !== null || child.signalCode !== null) {
      onExit(child.exitCode, child.signalCode);
    }
  });

export const waitForGuardedProcess = async (
  worker,
  {
    authorities,
    label,
    timeoutMs,
    termGraceMs = 2_000,
  }
) => {
  try {
    await Promise.race([
      waitForProcess(worker, { label, timeoutMs, termGraceMs }),
      ...authorities.map(({ child, label: authorityLabel }) =>
        waitForUnexpectedExit(child, authorityLabel)
      ),
    ]);
  } catch (error) {
    await stopProcess(worker, termGraceMs);
    throw error;
  }
};
