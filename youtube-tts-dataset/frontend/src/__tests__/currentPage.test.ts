import { currentPage } from "../App";

afterEach(() => window.history.replaceState(null, "", "/"));

test("pathname /studio selects studio when there is no hash", () => {
  window.history.replaceState(null, "", "/studio");
  expect(currentPage()).toBe("studio");
});

test("a hash overrides pathname /studio", () => {
  window.history.replaceState(null, "", "/studio#/jobs");
  expect(currentPage()).toBe("jobs");
});
