import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

export const fetchProducts = () => apiClient.get("/products/");

export const fetchForecast = (product, horizonMonths) =>
  apiClient.post("/forecast/", { product, horizon_months: horizonMonths });

export const fetchModelMetrics = () => apiClient.get("/model-metrics/");
