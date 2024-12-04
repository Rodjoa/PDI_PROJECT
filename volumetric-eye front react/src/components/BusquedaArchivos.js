import React, { useState } from 'react';

function BusquedaArchivos({ setView }) {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    uploadFile(selectedFile);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    setFile(droppedFile);
    uploadFile(droppedFile);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      alert(data.message);
    } catch (error) {
      console.error('Error al subir el archivo:', error);
    }
  };

  return (
    <div className="central-panel">
      <h2>Cargar Archivos</h2>
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        style={{
          border: '2px dashed #007bff',
          borderRadius: '8px',
          padding: '20px',
          textAlign: 'center',
          width: '100%',
          height: '150px',
          marginBottom: '20px',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          cursor: 'pointer',
        }}
      >
        {file ? <p>Archivo seleccionado: {file.name}</p> : <p>Arrastra tu archivo aquí o haz clic abajo para seleccionar</p>}
      </div>
      <input type="file" onChange={handleFileChange} />
      <button onClick={() => setView('previsualizacion')}>Vista Previa</button>
      <button onClick={() => setView('deteccion')}>Procesar</button>
      <button onClick={() => setView('home')}>Regresar</button>
    </div>
  );
}

export default BusquedaArchivos;
