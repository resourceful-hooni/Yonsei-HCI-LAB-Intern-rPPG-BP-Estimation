import { Helmet } from "react-helmet";

export default function MetaTags() {
  return (
    <Helmet>
      <meta property="og:title" content="VisiVital" />
      <meta property="og:description" content="rPPG 기반 건강 모니터링" />
      <meta property="og:image" content="/og-image.svg" />
      <meta property="og:type" content="website" />
      <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
      <link rel="apple-touch-icon" href="/favicon.svg" />
    </Helmet>
  );
}
