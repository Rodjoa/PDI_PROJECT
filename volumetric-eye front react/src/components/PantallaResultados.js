import React from 'react';

function PantallaResultados({ setView }) {
  const downloadResults = async () => {
    try {
      const response = await fetch('http://localhost:5000/download');
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'resultados.zip';
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (error) {
      console.error('Error al descargar el archivo:', error);
    }
  };

  const clearFiles = async () => {
    try {
      const response = await fetch('http://localhost:5000/clear', { method: 'DELETE' });
      const data = await response.json();
      alert(data.message);
    } catch (error) {
      console.error('Error al limpiar los archivos:', error);
    }
  };

  return (
    <div className="central-panel">
      <h2>Resultados</h2>
      <p>Aquí se muestran los resultados del procesamiento.</p>
      <button onClick={downloadResults}>Descargar Resultados</button>
      <button onClick={clearFiles}>Limpiar Archivos</button>
      <button onClick={() => setView('home')}>Regresar</button>
    </div>
  );
}

export default PantallaResultados;
