declare module "shiki/langs/*" {
  import type { LanguageRegistration } from "shiki/core";

  const language: LanguageRegistration;
  export default language;
}

declare module "shiki/themes/*" {
  import type { ThemeRegistration } from "shiki/core";

  const theme: ThemeRegistration;
  export default theme;
}
