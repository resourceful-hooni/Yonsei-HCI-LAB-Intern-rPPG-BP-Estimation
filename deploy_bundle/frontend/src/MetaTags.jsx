// React 19는 컴포넌트 안의 <meta>/<link>를 자동으로 <head>로 올려줌 (react-helmet 불필요)
export default function MetaTags() {
  return (
    <>
      <meta property="og:title" content="VisiVital" />
      <meta property="og:description" content="rPPG 기반 건강 모니터링" />
      <meta property="og:image" content="/og-image.svg" />
      <meta property="og:type" content="website" />
      <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
      <link rel="apple-touch-icon" href="/favicon.svg" />
    </>
  );
}
