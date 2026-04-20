export default function ProductSelector({ products, selected, onChange }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
        Software Product
      </label>
      <select
        value={selected}
        onChange={(e) => onChange(e.target.value)}
        className="px-4 py-2.5 rounded-lg border border-gray-300 bg-white text-gray-800
                   font-medium shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500
                   min-w-[180px] cursor-pointer"
      >
        {products.length > 0
          ? products.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name} ({p.current_licenses} seats)
              </option>
            ))
          : ["Jira", "Slack", "Zoom"].map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
      </select>
    </div>
  );
}
