# Code Execution Production Rollout

1. Confirm the dedicated code execution service is healthy on the private service node.
2. Set `CODE_EXECUTION_SERVICE_URL`, `CODE_EXECUTION_SERVICE_API_KEY`, `CODE_EXECUTION_ENABLED=true`, and `CODE_EXECUTION_DEFAULT_BACKEND=service` in `/etc/ultra/ultra-backend.env`.
3. Deploy the exact backend release snapshot to the app node.
4. Verify `https://<public-host>/v1/health`.
5. Run the service-backed smoke: `python3 scripts/smoke_pro_mode_code_execution.py --service-url <private-service-url> --api-key <private-token> --api-root https://<public-host>`.
6. Run the broader Opus smoke: `python3 scripts/smoke_pro_mode_opus.py`.
7. In a browser, log in with the admin test account, run one code-execution prompt that generates saved artifacts, then ask a follow-up in the same thread and confirm the answer reuses the prior measured context.

## Expected Behavior

- `execute_python_job` should route to the dedicated service backend by default.
- Hard computational prompts should upgrade from the generic tool workflow to the dedicated code-execution reasoning solver.
- Failed execution must fail closed: the final answer should explain the failure and omit unmeasured numeric outputs.
- Successful execution should mention the method actually used, key quantitative findings, produced artifact types, and limitations.
