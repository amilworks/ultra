<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>BisQue Login</title>
    <link rel="icon" type="image/png" href="${url.resourcesPath}/img/dna_cover.png" />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
    <link rel="stylesheet" href="${url.resourcesPath}/css/login.css" />
</head>
<body>
<main class="auth-screen">
    <section class="auth-screen-hero">
        <div class="auth-screen-hero-overlay">
            <div class="auth-screen-logo">
                <div class="auth-screen-logo-mark" aria-hidden="true">
                    <svg viewBox="0 0 128 128" fill="none">
                        <g stroke="currentColor" stroke-width="8" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M20 32V64A44 44 0 0 0 108 64V32" />
                            <path d="M34 58A16 16 0 0 1 64 48A16 16 0 0 0 94 58" />
                        </g>
                    </svg>
                </div>
                <span>BisQue Ultra</span>
            </div>
            <h1>Connect your BisQue account</h1>
            <p>
                Use the same credentials you use on BisQue. After sign-in, uploads, browsing,
                and tool calls run against your account.
            </p>
            <a href="${properties.bisqueClientUrl!'https://bisque.ece.ucsb.edu/client_service/'}" target="_blank" rel="noreferrer">Open BisQue client</a>
        </div>
    </section>

    <section class="auth-screen-form">
        <article class="auth-card">
            <header>
                <h2>Welcome back</h2>
                <p>Sign in with your BisQue username and password.</p>
            </header>

            <#if message?has_content>
                <div class="auth-error auth-error-${message.type}">${kcSanitize(message.summary)?no_esc}</div>
            </#if>

            <form id="kc-form-login" class="auth-form" action="${url.loginAction}" method="post">
                <#if !(usernameHidden?? && usernameHidden)>
                    <label class="auth-label" for="username">Username or email</label>
                    <input
                        id="username"
                        class="auth-input"
                        name="username"
                        type="text"
                        value="${(login.username!'')}"
                        autocomplete="username"
                        autofocus
                    />
                <#else>
                    <input type="hidden" name="username" value="${(login.username!'')}" />
                </#if>

                <label class="auth-label" for="password">Password</label>
                <input
                    id="password"
                    class="auth-input"
                    name="password"
                    type="password"
                    autocomplete="current-password"
                />

                <div class="auth-row">
                    <#if realm.rememberMe && !(usernameHidden?? && usernameHidden)>
                        <label class="auth-checkbox">
                            <input id="rememberMe" name="rememberMe" type="checkbox" <#if login?? && login.rememberMe??>checked</#if> />
                            <span>Remember me</span>
                        </label>
                    </#if>
                    <#if realm.resetPasswordAllowed>
                        <a class="auth-link" href="${url.loginResetCredentialsUrl}">Forgot password?</a>
                    </#if>
                </div>

                <input id="id-hidden-input" name="credentialId" type="hidden" <#if auth?? && auth.selectedCredential?has_content>value="${auth.selectedCredential}"</#if> />

                <button class="auth-submit" name="login" id="kc-login" type="submit">Sign in</button>

                <#if realm.registrationAllowed>
                    <p class="auth-register">Need an account? <a href="${url.registrationUrl}">Register</a></p>
                </#if>
            </form>

            <#if social?? && social.providers?? && (social.providers?size > 0)>
                <div class="auth-provider-wrap">
                    <p class="auth-divider"><span>Or continue with</span></p>
                    <div class="auth-provider-list">
                        <#list social.providers as p>
                            <a class="auth-provider-btn" href="${p.loginUrl}">
                                <span>${kcSanitize(p.displayName)?no_esc}</span>
                            </a>
                        </#list>
                    </div>
                </div>
            </#if>
        </article>
    </section>
</main>
</body>
</html>
