# Prerequisitos
**Hardware**:
* [CUDA compatible GPU](https://developer.nvidia.com/cuda-gpus)

**Software**:
* WSL2 in Windows 11
* Ubuntu 22.04 LTS en WSL2
* nodeJs
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* [Nvidia CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) **and post-installation actions** | **IMPORTANT**: After installation make sure `nvidia-smi` command works properly. If not, check GPU, nvidia drivers and CUDA version compability.

* [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

De cumplir con los prerequisitos al ejecutar el siguiente comando en la consola de ubuntu de wsl 
```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

De salir el siguiente mensaje, significa que el hardware y software se encuentran configurados correctamente
```bash
Thu Jul  4 16:03:13 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.42.06              Driver Version: 555.42.06      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GTX 1060 6GB    Off |   00000000:01:00.0  On |                  N/A |
|  0%   51C    P2             26W /  200W |     370MiB /   6144MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+

```

Si el contenedor acoplable no puede ver la GPU pero nvidia-smi funciona localmente, considere ejecutar:
```bash
sudo nano /etc/nvidia-container-runtime/config.toml
```

y ajustar la siguiente linea:
```bash
no-cgroups = false
```


# instalacion Contenedor biomasa en WSL2

una vez que se cumplan los prerequisitos, se puede instalar el contenedor de la siguiente manera:

1) asegure de tener el sistema actualizado:
    ```bash
    sudo apt-get update && sudo apt-get upgrade
    ```

2) Clone el repositorio en el directorio $HOME e ingrese en el:
    ```bash
    cd
    git clone git@gitlab.com:wildsense/fondef-aquarov/cv-estimation-algorithms/biomass_estimation_stable.git
    cd ~/biomass_estimation_stable
    ```

3) actualizar sub-repositorios:
    
    ```bash
    git submodule update --init
    ```
4) crear la imagen de Docker mass-estimation de forma local (**puede tomar varios minutos**):
    ```bash
    docker build -t mass-estimation:latest ./Documentation/Docker
    ```

5) Create configuration file specific to your machine:
   ```bash
    cat << EOF > ~/biomass_estimation_stable/Scripts/conf.env
    INPUT_PATH=/su/ruta/a/carpeta/input_video
    OUTPUT_PATH=/su/ruta/a/carpeta/output_video
    WORK_MODE=F    # Defines if the machine will process the pending videos in FIFO or LIFO order. FIFO default
    USERNAME=*
    CENTER=
    EOF
    ```
6) cloanr nuestro repositorio de codigo en la carpeta $HOME
    ```bash
    cd
    git clone https://github.com/Rodjoa/PDI_PROJECT.git
    ```
7) para usar nuestro codigo debe copiar el archivo foam_fish_debugguerv3.py en la ruta ~/biomass_estimation_stable/BiomassEstimator/Utils/FoamFishDebugguer
    ```bash
    cp ~/PDI_PROJECT/codigo/foam_fish_debugguerv3.py ~/biomass_estimation_stable/BiomassEstimator/Utils/FoamFishDebugguer
    ```
    
8) verificar que el contenedor puede iniciar correctamente:
    ```bash
    bash Scripts/StartContainer.sh 
    ```
9) Una vez que el contenedor esta en ejecucion se puede conectar a el dese VScode como coneccion remota y aceder al a la ruta
    ```bash
    /home/mass_estimation/BiomassEstimator/Utils/FoamFishDebugguer/foam_fish_debugguerv3.py
    ```
    para ejecutar nuestro codigo.

10) para usar la plataforma react deve usarse el siguiente comando:
    ```bash
    bash npm install
    ```
11)
    ```bash
    bash npm start
    ```

# Biomass Debugging


Luego, en la parte inferior izquiera de la pantalla deberia tener disponible la opción "open remote window" (las dos flechitas que se apuntan entre ellas). Al presionarlo, seleccione la opción "attach to running container" y luego seleccione mass-estimation. Asi quedara vinculado correctamente dentro de vc code el container y se podrá codear.

Finalmente, abra la carpeta **/home/mass_estimation/** dentro del contenedor a traves de VS code para ver el repositorio del proyecto completo.

# Anexos
Se ha modificado el dataset salmons ya que habia un error con el estandar COCO para ser leido por los scripts de felo. Esto debe considerarse a la hora de volver a agregarlos.

 - Anexo 1: Si no tiene instalado docker, debe instalarlo, incluyendo las dependencias de NVIDIA para el funcionamiento de containers con tarjeta de video. Siga la instalación de https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker en ubuntu, que le recomendará el lugar indicado donde instalar docker y como instalar y activar el soporte de NVIDIA.

 - Anexo 2: Si tiene problemas con permisos sobre docker (esto es, ejeucción de comandos sudo cuando no lo debería pedir) debe darle permisos a todos los usuarios para instalar docker. Vease https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue por ejemplo. 

 - Anexo 3: Si tiene problemas con la instalación de nvidia-container-toolkit, puede ser que no tenga el repositorio de NVIDIA agregado a su sistema. Siga las instrucciones de https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker para agregar el repositorio y volver a intentar la instalación.