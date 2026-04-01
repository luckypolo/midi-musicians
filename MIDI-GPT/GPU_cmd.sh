cd ~
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8

export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$PWD/libraries/libtorch/lib:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="12.0"

which nvcc
nvcc --version

cd ~/Desktop/M1/T4/IFT_6289-Natural_Langage_Processing/Project/Code/midigpt_workspace/MIDI-GPT-inference-gpu

cat > patch_build.py <<'PY'
from pathlib import Path
import re

p = Path("create_python_library.sh")
s = p.read_text()

s_new, n1 = re.subn(
    r'https://download\.pytorch\.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-[^"]+',
    'https://download.pytorch.org/libtorch/test/cu128/libtorch-cxx11-abi-shared-with-deps-latest.zip',
    s
)

if 'CMAKE_CUDA_ARCHITECTURES=120' not in s_new:
    s_new, n2 = re.subn(
        r'if ! \$no_torch; then',
        'if $cuda; then cmake_flags="$cmake_flags -DCMAKE_CUDA_ARCHITECTURES=120"; fi if ! $no_torch; then',
        s_new,
        count=1
    )
else:
    n2 = 1

p.write_text(s_new)

if n1 == 0:
    raise SystemExit("No cu118 URL was replaced.")
if n2 == 0:
    raise SystemExit("Could not insert CMAKE_CUDA_ARCHITECTURES=120.")
print("Patched create_python_library.sh successfully.")
PY

python3 patch_build.py

python3 - <<'PY'
from pathlib import Path

p = Path("CMakeLists.txt")
s = p.read_text()

needle = 'add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libraries/pybind11)'
replacement = (
    'set(PYBIND11_FINDPYTHON ON)\n'
    'find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)\n'
    'add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libraries/pybind11)'
)

if 'Development.Module REQUIRED' not in s:
    s = s.replace(needle, replacement, 1)
    p.write_text(s)

print("Patched CMakeLists.txt")
PY

rm -rf libraries/libtorch
rm -rf python_lib
rm -rf libraries/protobuf/build

bash create_python_library.sh --cuda --env_name venv

source ../venv/bin/activate
cd python_lib
python -c "import midigpt; print(midigpt.__file__)"

cd ../python_scripts_for_testing
python pythoninferencetest.py \
  --midi mtest.mid \
  --ckpt ../models/EXPRESSIVE_ENCODER_RES_1920_12_GIGAMIDI_CKPT_150K.pt \
  --out ../output_gpu.mid

python LLM_enhance.py \
  --midi mtest.mid \
  --ckpt ../models/EXPRESSIVE_ENCODER_RES_1920_12_GIGAMIDI_CKPT_150K.pt \
  --out ../output_gpu.mid