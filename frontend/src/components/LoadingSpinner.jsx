export default function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-24">
      <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent" />
      <span className="ml-4 text-gray-500 text-lg">Loading forecast...</span>
    </div>
  );
}
