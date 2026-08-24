import '@testing-library/jest-dom/vitest'

// jsdomはscrollIntoViewを実装していないため、ChatMessageListの自動スクロールが
// テスト実行時に例外を投げないようダミー実装を用意する。
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {}
}
