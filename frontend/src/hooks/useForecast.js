import { useState, useEffect, useCallback } from "react";
import { fetchForecast, fetchProducts } from "../api/client";

export function useForecast() {
  const [products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState("Jira");
  const [horizon, setHorizon] = useState(6);
  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchProducts()
      .then((res) => setProducts(res.data))
      .catch(() => {});
  }, []);

  const loadForecast = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const { data } = await fetchForecast(selectedProduct, horizon);
      setForecastData(data);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Failed to load forecast");
    } finally {
      setLoading(false);
    }
  }, [selectedProduct, horizon]);

  useEffect(() => {
    loadForecast();
  }, [loadForecast]);

  return {
    products,
    selectedProduct,
    setSelectedProduct,
    horizon,
    setHorizon,
    forecastData,
    loading,
    error,
    refresh: loadForecast,
  };
}
