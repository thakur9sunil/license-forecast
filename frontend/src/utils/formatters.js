import dayjs from "dayjs";

export const formatDate = (dateStr) => dayjs(dateStr).format("MMM D, YYYY");
export const formatMonthYear = (dateStr) => dayjs(dateStr).format("MMM YY");
export const formatNumber = (n) => Math.round(n).toLocaleString();
export const formatPercent = (n) =>
  `${n > 0 ? "+" : ""}${Number(n).toFixed(1)}%`;

export const RECOMMENDATION_LABELS = {
  buy_more: "Buy More Licenses",
  reduce: "Reduce Licenses",
  hold: "Hold Current",
};

export const RECOMMENDATION_COLORS = {
  buy_more: { bg: "bg-red-50", text: "text-red-700", border: "border-red-200", dot: "bg-red-500" },
  reduce: { bg: "bg-green-50", text: "text-green-700", border: "border-green-200", dot: "bg-green-500" },
  hold: { bg: "bg-blue-50", text: "text-blue-700", border: "border-blue-200", dot: "bg-blue-500" },
};

export const TREND_ICONS = {
  increasing: "↑",
  decreasing: "↓",
  stable: "→",
};

export const TREND_COLORS = {
  increasing: "text-orange-600",
  decreasing: "text-blue-600",
  stable: "text-gray-600",
};
