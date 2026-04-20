const OPTIONS = [
  { value: 3, label: "3 Months" },
  { value: 6, label: "6 Months" },
  { value: 12, label: "12 Months" },
];

export default function HorizonSelector({ horizon, onChange }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
        Forecast Horizon
      </label>
      <div className="flex rounded-lg border border-gray-300 overflow-hidden shadow-sm">
        {OPTIONS.map(({ value, label }) => (
          <button
            key={value}
            onClick={() => onChange(value)}
            className={`px-4 py-2.5 text-sm font-medium transition-colors
              ${
                horizon === value
                  ? "bg-blue-600 text-white"
                  : "bg-white text-gray-600 hover:bg-gray-50"
              }
              ${value !== 3 ? "border-l border-gray-300" : ""}`}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}
