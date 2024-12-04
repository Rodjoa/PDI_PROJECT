const express = require('express');
const multer = require('multer');
const archiver = require('archiver');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 5000;

// Configurar almacenamiento para los archivos cargados
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = 'uploads';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir);
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, file.originalname); // Mantener el nombre original del archivo
  },
});
const upload = multer({ storage });

// Habilitar CORS
app.use(cors());

// Ruta para cargar archivos
app.post('/upload', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).send('No se envió ningún archivo.');
  }
  res.send({ message: 'Archivo cargado exitosamente.', filename: req.file.originalname });
});

// Ruta para descargar los archivos cargados como un archivo ZIP
app.get('/download', (req, res) => {
  const zipFile = path.join(__dirname, 'resultados.zip');
  const output = fs.createWriteStream(zipFile);
  const archive = archiver('zip', { zlib: { level: 9 } });

  output.on('close', () => {
    res.download(zipFile, () => {
      fs.unlinkSync(zipFile); // Eliminar el archivo ZIP después de la descarga
    });
  });

  archive.on('error', (err) => {
    throw err;
  });

  archive.pipe(output);

  // Agregar archivos cargados al archivo ZIP
  const filesDir = path.join(__dirname, 'uploads');
  if (fs.existsSync(filesDir)) {
    fs.readdirSync(filesDir).forEach((file) => {
      archive.file(path.join(filesDir, file), { name: file });
    });
  } else {
    res.status(404).send('No hay archivos para descargar.');
  }

  archive.finalize();
});

// Ruta para limpiar los archivos cargados
app.delete('/clear', (req, res) => {
  const filesDir = path.join(__dirname, 'uploads');
  if (fs.existsSync(filesDir)) {
    fs.readdirSync(filesDir).forEach((file) => fs.unlinkSync(path.join(filesDir, file)));
  }
  res.send({ message: 'Archivos eliminados correctamente.' });
});

// Iniciar el servidor
app.listen(PORT, () => {
  console.log(`Servidor corriendo en http://localhost:${PORT}`);
});
