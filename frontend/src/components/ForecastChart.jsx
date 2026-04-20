import {
  ComposedChart,
  Line,
  Area,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import dayjs from "dayjs";
import { formatDate, formatMonthYear, formatNumber } from "../utils/formatters";

function buildChartData(historical, forecast) {
  const hist = historical.map((p) => ({
    ds: p.ds,
    actual: p.y,
    ciLow: undefined,
    ciHigh: undefined,
    forecast: undefined,
  }));

  const fcast = forecast.map((p) => ({
    ds: p.ds,
    actual: undefined,
    ciLow: p.yhat_lower,
    ciHigh: p.yhat_upper,
    forecast: p.yhat,
  }));

  return [...hist, ...fcast];
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;

  const dataPoint = payload[0]?.payload || {};
  const isHistorical = dataPoint.actual !== undefined;

  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-3 text-sm">
      <p className="font-semibold text-gray-700 mb-1">{formatDate(label)}</p>
      {isHistorical ? (
        <p className="text-blue-600">
          Actual: <strong>{formatNumber(dataPoint.actual)}</strong>
        </p>
      ) : (
        <>
          <p className="text-orange-500">
            Forecast: <strong>{formatNumber(dataPoint.forecast)}</strong>
          </p>
          <p className="text-gray-400 text-xs">
            Range: {formatNumber(dataPoint.ciLow)} – {formatNumber(dataPoint.ciHigh)}
          </p>
        </>
      )}
    </div>
  );
}

export default function ForecastChart({ data }) {
  const { historical, forecast, renewal_date, product } = data;
  const chartData = buildChartData(historical, forecast);

  const allValues = [
    ...historical.map((p) => p.y),
    ...forecast.map((p) => p.yhat_upper),
    ...forecast.map((p) => p.yhat_lower),
  ].filter(Boolean);
  const yMin = Math.max(0, Math.floor(Math.min(...allValues) * 0.92));
  const yMax = Math.ceil(Math.max(...allValues) * 1.08);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-800">
          {product} — License Usage Forecast
        </h2>
        <span className="text-xs text-gray-400">
          Renewal: {formatDate(renewal_date)}
        </span>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />

          <XAxis
            dataKey="ds"
            tickFormatter={formatMonthYear}
            tick={{ fontSize: 12, fill: "#6b7280" }}
          />
          <YAxis
            domain={[yMin, yMax]}
            tickFormatter={(v) => formatNumber(v)}
            tick={{ fontSize: 12, fill: "#6b7280" }}
            label={{
              value: "Licenses",
              angle: -90,
              position: "insideLeft",
              offset: 10,
              style: { fontSize: 12, fill: "#9ca3af" },
            }}
            width={75}
          />

          <Tooltip content={<CustomTooltip />} />

          <Legend
            wrapperStyle={{ fontSize: 13, paddingTop: 12 }}
            formatter={(value) => {
              const labels = {
                actual: "Historical Usage",
                forecast: "Forecast",
                ciHigh: "Confidence Interval",
              };
              return labels[value] || value;
            }}
          />

          {/* Confidence interval band — ciLow as transparent base, ciHigh as fill */}
          <Area
            dataKey="ciLow"
            stroke="none"
            fill="transparent"
            legendType="none"
            isAnimationActive={false}
          />
          <Area
            dataKey="ciHigh"
            stroke="none"
            fill="#fed7aa"
            fillOpacity={0.5}
            name="ciHigh"
            isAnimationActive={false}
          />

          {/* Historical actual line */}
          <Line
            dataKey="actual"
            stroke="#3b82f6"
            strokeWidth={2.5}
            dot={{ r: 4, fill: "#3b82f6", strokeWidth: 0 }}
            activeDot={{ r: 6 }}
            connectNulls={false}
            name="actual"
            isAnimationActive={false}
          />

          {/* Forecast dashed line */}
          <Line
            dataKey="forecast"
            stroke="#f97316"
            strokeWidth={2.5}
            strokeDasharray="6 3"
            dot={{ r: 4, fill: "#f97316", strokeWidth: 0 }}
            activeDot={{ r: 6 }}
            connectNulls={false}
            name="forecast"
            isAnimationActive={false}
          />

          {/* Renewal date marker */}
          <ReferenceLine
            x={renewal_date}
            stroke="#ef4444"
            strokeDasharray="4 4"
            strokeWidth={2}
            label={{
              value: "Renewal",
              position: "top",
              fill: "#ef4444",
              fontSize: 12,
              fontWeight: 600,
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
