import {
  formatDate,
  formatNumber,
  formatPercent,
  RECOMMENDATION_LABELS,
  RECOMMENDATION_COLORS,
  TREND_ICONS,
  TREND_COLORS,
} from "../utils/formatters";

function Card({ title, value, subtitle, valueClass = "" }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">
        {title}
      </p>
      <p className={`text-3xl font-bold text-gray-800 ${valueClass}`}>{value}</p>
      {subtitle && <p className="text-sm text-gray-500 mt-1">{subtitle}</p>}
    </div>
  );
}

export default function SummaryCards({ data }) {
  const {
    current_usage,
    predicted_at_renewal,
    percent_change,
    renewal_date,
    recommendation,
    recommendation_detail,
    trend_direction,
    model_version,
  } = data;

  const recColors = RECOMMENDATION_COLORS[recommendation] || RECOMMENDATION_COLORS.hold;
  const changePositive = percent_change > 0;
  const changeColor = changePositive ? "text-red-600" : percent_change < 0 ? "text-green-600" : "text-gray-600";

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <Card
        title="Current Usage"
        value={formatNumber(current_usage)}
        subtitle="licenses in use (last month)"
      />
      <Card
        title="Predicted at Renewal"
        value={formatNumber(predicted_at_renewal)}
        subtitle={`Due: ${formatDate(renewal_date)}`}
        valueClass={changePositive ? "text-red-600" : "text-green-600"}
      />
      <Card
        title="Projected Change"
        value={
          <span className={changeColor}>
            {TREND_ICONS[trend_direction]} {formatPercent(percent_change)}
          </span>
        }
        subtitle={`Trend: ${trend_direction}`}
      />
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">
          Recommendation
        </p>
        <span
          className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-semibold
            border ${recColors.bg} ${recColors.text} ${recColors.border}`}
        >
          <span className={`w-2 h-2 rounded-full ${recColors.dot}`} />
          {RECOMMENDATION_LABELS[recommendation]}
        </span>
        <p className="text-xs text-gray-500 mt-2 leading-relaxed line-clamp-3">
          {recommendation_detail}
        </p>
        <p className="text-xs text-gray-300 mt-2">Model: {model_version}</p>
      </div>
    </div>
  );
}
