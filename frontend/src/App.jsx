import { useForecast } from "./hooks/useForecast";
import ProductSelector from "./components/ProductSelector";
import HorizonSelector from "./components/HorizonSelector";
import ForecastChart from "./components/ForecastChart";
import SummaryCards from "./components/SummaryCards";
import LoadingSpinner from "./components/LoadingSpinner";

export default function App() {
  const {
    products,
    selectedProduct,
    setSelectedProduct,
    horizon,
    setHorizon,
    forecastData,
    loading,
    error,
    refresh,
  } = useForecast();

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              License Usage Forecast
            </h1>
            <p className="text-sm text-gray-500 mt-0.5">
              ML-powered renewal recommendations · Prophet + SARIMA
            </p>
          </div>
          <button
            onClick={refresh}
            disabled={loading}
            className="px-4 py-2 text-sm font-medium text-blue-600 border border-blue-200
                       rounded-lg hover:bg-blue-50 transition-colors disabled:opacity-50"
          >
            ↺ Refresh
          </button>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Controls */}
        <div className="flex flex-wrap gap-4 mb-8">
          <ProductSelector
            products={products}
            selected={selectedProduct}
            onChange={setSelectedProduct}
          />
          <HorizonSelector horizon={horizon} onChange={setHorizon} />
        </div>

        {/* Loading */}
        {loading && <LoadingSpinner />}

        {/* Error */}
        {error && !loading && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-center">
            <p className="text-red-700 font-semibold">Failed to load forecast</p>
            <p className="text-red-500 text-sm mt-1">{error}</p>
            <p className="text-gray-400 text-xs mt-3">
              Make sure the backend is running at{" "}
              <code className="font-mono">http://localhost:8000</code> and models
              have been trained.
            </p>
          </div>
        )}

        {/* Results */}
        {forecastData && !loading && (
          <>
            <SummaryCards data={forecastData} />
            <ForecastChart data={forecastData} />

            {/* Model info footer */}
            <div className="text-xs text-gray-400 text-right mt-2">
              Generated: {new Date(forecastData.generated_at).toLocaleString()} ·
              Model v{forecastData.model_version} ·
              {forecastData.horizon_months}-month horizon
            </div>
          </>
        )}
      </main>
    </div>
  );
}
