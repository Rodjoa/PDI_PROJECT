import React from 'react';

function Previsualizacion({ setView }) {
  return (
    <div className="central-panel">
      <h2>Previsualización</h2>
      <p>Aquí se muestra una vista previa de la imagen o video seleccionado.</p>
      <button onClick={() => setView('deteccion')}>Procesar</button>
      <button onClick={() => setView('home')}>Regresar</button>
    </div>
  );
}

export default Previsualizacion;
