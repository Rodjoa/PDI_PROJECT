import React, { useState } from 'react';

function BusquedaArchivos({ setView }) {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  return (
    <div className="central-panel">
      <h2>Cargar Archivos</h2>
      <input type="file" onChange={handleFileChange} />
      {file && <p>Archivo seleccionado: {file.name}</p>}
      <button onClick={() => setView('previsualizacion')}>Vista Previa</button>
      <button onClick={() => setView('deteccion')}>Procesar</button>
      <button onClick={() => setView('home')}>Regresar</button>
    </div>
  );
}

export default BusquedaArchivos;
