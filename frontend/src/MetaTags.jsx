import { Helmet } from "react-helmet";

export default function MetaTags() {
  return (
    <Helmet>
      <meta property="og:title" content="VisiVital" />
      <meta property="og:description" content="Contactless health monitoring with rPPG (reference blood pressure / glucose)" />
      <meta property="og:image" content="/og-image.svg" />
      <meta property="og:type" content="website" />
      <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
      <link rel="apple-touch-icon" href="/favicon.svg" />
    </Helmet>
  );
}
