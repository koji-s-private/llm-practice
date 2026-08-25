import '@testing-library/jest-dom/vitest'

// jsdomはscrollIntoViewを実装していないため、ChatMessageListの自動スクロールが
// 参照するとTypeErrorになる。テスト環境でのみno-opを補う。
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {}
}
