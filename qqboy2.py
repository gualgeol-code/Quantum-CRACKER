#i realy hope you get me some Donation for the Quantum project_ 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai /////
import cupy as cp  # CUDA-accelerated operations using CuPy
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.controlflow.break_loop import BreakLoopPlaceholder
from qiskit.circuit.library import ZGate, MCXGate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session, Options
from qiskit.primitives import SamplerResult
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from collections import Counter
from Crypto.Hash import RIPEMD160, SHA256  # Import from pycryptodome
from ecdsa import SigningKey, SECP256k1
from qiskit.quantum_info import Statevector
from bitarray import bitarray
import random
import time
import hashlib
import base58
import numpy as np
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Operator
import math

# Load IBMQ account using QiskitRuntimeService dengan instance yang spesifik
QiskitRuntimeService.save_account(
    channel='ibm_quantum_platform',
    token='yY74ekU5S_x2GpDawPqq0KZpbkOcikB9KAZs-jB1mdy-',
    overwrite=True,
    set_as_default=True,
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/8b1f2c19555e44cca6b21ce9e0fdfcfe:6a63c2ec-38c9-4177-b23c-105bdb444891::"
)

# Load the service dengan instance yang spesifik
service = QiskitRuntimeService(
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/7d8b6f65e3bb4d76ad7af2f598cc70ca:a4a8e2da-5225-42c3-bbaf-34dba6dd020e::"
)

SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

def apply_qft(circuit, num_qubits):
    qft = QFT(num_qubits)
    circuit.append(qft, range(num_qubits))
    
    for j in range(num_qubits):
        circuit.h(j)
        for k in range(j + 1, num_qubits):
            circuit.cp(np.pi / (2 ** (k - j)), j, k)
    
    for j in range(num_qubits // 2):
        circuit.swap(j, num_qubits - j - 1)

def private_key_to_compressed_address(private_key_hex):
    print(f"Converting private key {private_key_hex} to Bitcoin address...")
    private_key_bytes = bytes.fromhex(private_key_hex)
    sk = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
    vk = sk.verifying_key
    public_key_bytes = vk.to_string()
    x_coord = public_key_bytes[:32]
    y_coord = public_key_bytes[32:]
    prefix = b'\x02' if int.from_bytes(y_coord, 'big') % 2 == 0 else b'\x03'
    compressed_public_key = prefix + x_coord

    sha256_pk = hashlib.sha256(compressed_public_key).digest()
    ripemd160 = RIPEMD160.new()
    ripemd160.update(sha256_pk)
    hashed_public_key = ripemd160.digest()

    network_byte = b'\x00' + hashed_public_key
    sha256_first = hashlib.sha256(network_byte).digest()
    sha256_second = hashlib.sha256(sha256_first).digest()
    checksum = sha256_second[:4]

    binary_address = network_byte + checksum
    bitcoin_address = base58.b58encode(binary_address).decode('utf-8')
    print(f"Generated Bitcoin address: {bitcoin_address}")
    return bitcoin_address

def grover_oracle(circuit, private_key_qubits, public_key_x, g_x, g_y, p, ancilla):
    circuit.barrier()
    num_qubits = 125
    target_state = random.randint(0x4000000000000000000000000000000000, 0x7fffffffffffffffffffffffffffffffff)
    computed_x, _ = scalar_multiplication(target_state, g_x, g_y, p)

    if computed_x == public_key_x:
        circuit.x(ancilla)

    circuit.barrier()

def mod_inverse(a, p):
    if a == 0:
        return 0
    lm, hm = 1, 0
    low, high = a % p, p
    while low > 1:
        ratio = high // low
        nm, new_low = hm - lm * ratio, high - low
        lm, low, hm, high = nm, new_low, lm, low
    return lm % p

def point_addition(x1, y1, x2, y2, p):
    if x1 == x2 and y1 == y2:
        return point_doubling(x1, y1, p)
    lam = ((y2 - y1) * mod_inverse(x2 - x1, p)) % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return x3, y3

def point_doubling(x1, y1, p):
    lam = ((3 * x1 * x1) * mod_inverse(2 * y1, p)) % p
    x3 = (lam * lam - 2 * x1) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return x3, y3

def scalar_multiplication(k, x, y, p):
    x_res, y_res = x, y
    k_bin = bin(k)[2:]
    for bit in k_bin[1:]:
        x_res, y_res = point_doubling(x_res, y_res, p)
        if bit == '1':
            x_res, y_res = point_addition(x_res, y_res, x, y, p)
    return x_res, y_res

def binary_to_hex(bin_key):
    bin_key = bin_key.zfill(128)
    return hex(int(bin_key, 2))[2:].zfill(64)

def quantum_brute_force(public_key_x: int, g_x: int, g_y: int, p: int, min_range: int, max_range: int):
    if max_range <= min_range:
        raise ValueError("max_range must be greater than min_range.")

    target_address = '16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v'
    quantum_registers = 100
    num_ancillas = 8
    num_iterations = 5  # Kurangi untuk testing, naikkan nanti

    print("Mencari backend yang tersedia...")
    try:
        backends = service.backends()
        print(f"Backends yang tersedia: {[b.name for b in backends]}")
        
        # Untuk testing, gunakan simulator jika ada
        simulator_found = False
        backend = None
        
        for b in backends:
            if b.configuration().simulator:
                backend = b
                simulator_found = True
                print(f"Menggunakan simulator: {backend.name}")
                break
        
        if not simulator_found:
            # Jika tidak ada simulator, cari hardware dengan qubit terbanyak
            hardware_backends = [b for b in backends if not b.configuration().simulator]
            if hardware_backends:
                # Cari yang memiliki pending jobs paling sedikit
                backend = min(hardware_backends, key=lambda b: b.status().pending_jobs)
                print(f"Menggunakan hardware: {backend.name} dengan {backend.configuration().n_qubits} qubit")
            else:
                raise Exception("Tidak ada backend yang tersedia")
            
    except Exception as e:
        print(f"Error mencari backend: {e}")
        return None

    attempt = 0
    
    while True:
        attempt += 1
        print(f"\n{'='*60}")
        print(f"Attempt {attempt} - Using backend: {backend.name}")
        print(f"{'='*60}")

        circuit = QuantumCircuit(quantum_registers + num_ancillas, quantum_registers)
        ancilla_register = QuantumRegister(num_ancillas, name='ancilla')
        circuit.add_register(ancilla_register)
        print("Quantum circuit initialized.")

        circuit.h(range(quantum_registers))
        print("Hadamard gates applied.")

        print(f"Applying Grover's iterations ({num_iterations} iterations).")
        for _ in range(num_iterations):
            grover_oracle(circuit, range(quantum_registers), public_key_x, g_x, g_y, p, ancilla_register[0])

        circuit.measure(range(quantum_registers), range(quantum_registers))
        print("Measurement operation added to circuit.")

        print("Transpiling the circuit for the selected backend...")
        transpiled_circuit = transpile(circuit, backend=backend, optimization_level=1)
        print(f"Circuit transpiled. Depth: {transpiled_circuit.depth()}")

        # =========== PERBAIKAN UTAMA: Menggunakan SamplerV2 dengan mode=backend ===========
        print("\nMenyiapkan SamplerV2 sesuai API terbaru dengan mode=backend...")

        try:
            # PERUBAHAN UTAMA: Gunakan mode=backend, bukan backend=backend
            sampler = SamplerV2(mode=backend)  # <-- PERUBAHAN DI SINI
            print(f"SamplerV2 created successfully with mode={backend.name}")

            # 2. JALANKAN CIRCUIT DENGAN SHOTS
            # Format PUB (Primitive Unified Blob) yang benar adalah:
            # (circuit, parameter_values, shots)
            # Karena circuit Anda tidak punya parameter, gunakan None untuk parameter_values.
            shots = 5000
            pub = (transpiled_circuit, None, shots)  # <-- PERHATIKAN FORMAT INI

            print(f"Running circuit with {shots} shots...")
            job = sampler.run([pub])  # Kirim sebagai list dari PUBs

            job_id = job.job_id()
            print(f"Job submitted successfully. Job ID: {job_id}")

            print("Waiting for job results...")
            job.wait_for_final_state()

            # 3. DAPATKAN DAN PROSES HASIL DENGAN CARA YANG BENAR
            result = job.result()
            print("Job completed successfully")

            # EKSTRAK COUNTS DARI HASIL - CARA YANG BENAR UNTUK API TERBARU
            counts = {}
            try:
                # Untuk SamplerV2, hasil biasanya dalam bentuk DataBin
                # Coba akses data counts dengan cara yang benar
                if hasattr(result[0].data, 'c') and result[0].data.c is not None:
                    counts_data = result[0].data.c
                    
                    # PERBAIKAN: Gunakan .get_counts() jika tersedia
                    if hasattr(counts_data, 'get_counts'):
                        counts_dict = counts_data.get_counts()
                        for bitstring, count in counts_dict.items():
                            # Pastikan bitstring memiliki panjang yang benar
                            if len(bitstring) < quantum_registers:
                                bitstring = bitstring.zfill(quantum_registers)
                            elif len(bitstring) > quantum_registers:
                                bitstring = bitstring[:quantum_registers]
                            
                            counts[bitstring] = count
                        print(f"Got {len(counts)} unique measurement results using get_counts()")
                    
                    # Alternatif: jika counts_data adalah array-like
                    elif hasattr(counts_data, '__array__'):
                        # Coba akses data dengan cara yang benar untuk array
                        try:
                            # Gunakan .tolist() atau akses langsung
                            if hasattr(counts_data, 'tolist'):
                                data_list = counts_data.tolist()
                                for i, count_val in enumerate(data_list):
                                    if count_val > 0:
                                        bin_key = format(i, f'0{quantum_registers}b')
                                        counts[bin_key] = int(count_val)
                                print(f"Got {len(counts)} unique measurement results using tolist()")
                            else:
                                # Akses sebagai array numpy
                                import numpy as np
                                arr = np.array(counts_data)
                                for i in range(len(arr)):
                                    if arr[i] > 0:
                                        bin_key = format(i, f'0{quantum_registers}b')
                                        counts[bin_key] = int(arr[i])
                                print(f"Got {len(counts)} unique measurement results using numpy array")
                        except Exception as array_error:
                            print(f"Error processing array data: {array_error}")
                            # Coba metode sederhana
                            try:
                                for i in range(len(counts_data)):
                                    count_val = counts_data[i]
                                    if count_val > 0:
                                        bin_key = format(i, f'0{quantum_registers}b')
                                        counts[bin_key] = int(count_val)
                                print(f"Got {len(counts)} unique measurement results (direct access)")
                            except:
                                print("Could not process counts data with any method")
                    
                    else:
                        print("Unknown data format, trying basic iteration...")
                        # Coba proses sebagai iterable
                        try:
                            for i, count_val in enumerate(counts_data):
                                if count_val > 0:
                                    bin_key = format(i, f'0{quantum_registers}b')
                                    counts[bin_key] = int(count_val)
                            print(f"Got {len(counts)} unique measurement results")
                        except Exception as e:
                            print(f"Error in basic iteration: {e}")
                            # Coba tampilkan struktur data untuk debugging
                            print(f"Data type: {type(counts_data)}")
                            if hasattr(counts_data, 'shape'):
                                print(f"Data shape: {counts_data.shape}")
                
                else:
                    print("No counts data found in result")
                    # Coba akses data dengan cara lain
                    print(f"Available data attributes: {dir(result[0].data)}")
                    
            except Exception as data_error:
                print(f"Error extracting counts data: {data_error}")
                import traceback
                traceback.print_exc()

            # Cek setiap hasil jika ada data
            if counts:
                print(f"Processing {len(counts)} measurement results...")
                for bin_key, count in counts.items():
                    if len(bin_key) < quantum_registers:
                        bin_key = bin_key.ljust(quantum_registers, '0')

                    private_key_hex = binary_to_hex(bin_key)
                    compressed_address = private_key_to_compressed_address(private_key_hex)

                    if compressed_address == target_address:
                        print(f"\n{'!'*60}")
                        print(f"SUCCESS! Found matching private key!")
                        print(f"Private Key: {private_key_hex}")
                        print(f"Bitcoin Address: {compressed_address}")
                        print(f"{'!'*60}")

                        with open('boomqft.txt', 'w') as f:
                            f.write(f"Found private key: {private_key_hex}\n")
                            f.write(f"Corresponding Bitcoin address: {compressed_address}\n")

                        return private_key_hex
            else:
                print("No counts data available to check")

            print("No matching key found in this attempt")

        except Exception as e:
            print(f"Error in quantum execution: {e}")
            print("Please check the Qiskit IBM Runtime version and API documentation.")
            import traceback
            traceback.print_exc()
            
            # Coba pendekatan alternatif jika mode=backend juga gagal
            print("\nTrying alternative approach with Session...")
            try:
                with Session(service=service, backend=backend) as session:
                    sampler = SamplerV2(session=session)
                    print("SamplerV2 created successfully with Session")
                    
                    # Jalankan circuit dengan format yang sama
                    shots = 5000
                    pub = (transpiled_circuit, None, shots)
                    print(f"Running circuit with {shots} shots...")
                    job = sampler.run([pub])
                    
                    job_id = job.job_id()
                    print(f"Job submitted successfully. Job ID: {job_id}")
                    
                    job.wait_for_final_state()
                    result = job.result()
                    print("Job completed successfully")
                    
                    # Proses hasil dengan cara yang sama
                    counts = {}
                    try:
                        if hasattr(result[0].data, 'c') and result[0].data.c is not None:
                            counts_data = result[0].data.c
                            
                            if hasattr(counts_data, 'get_counts'):
                                counts = counts_data.get_counts()
                                print(f"Got {len(counts)} measurement results using get_counts()")
                    except:
                        print("Could not extract counts from session result")
                    
                    if counts:
                        for bin_key, count in counts.items():
                            if len(bin_key) < quantum_registers:
                                bin_key = bin_key.ljust(quantum_registers, '0')

                            private_key_hex = binary_to_hex(bin_key)
                            compressed_address = private_key_to_compressed_address(private_key_hex)

                            if compressed_address == target_address:
                                print(f"\n{'!'*60}")
                                print(f"SUCCESS! Found matching private key!")
                                print(f"Private Key: {private_key_hex}")
                                print(f"Bitcoin Address: {compressed_address}")
                                print(f"{'!'*60}")

                                with open('boomqft.txt', 'w') as f:
                                    f.write(f"Found private key: {private_key_hex}\n")
                                    f.write(f"Corresponding Bitcoin address: {compressed_address}\n")

                                return private_key_hex
                    
                    print("No matching key found in this attempt")
                    
            except Exception as session_error:
                print(f"Session approach also failed: {session_error}")
                return None
        
        # Jika menggunakan hardware, kurangi frekuensi attempts
        if not backend.configuration().simulator:
            print("Waiting 30 seconds before next attempt...")
            time.sleep(30)
        else:
            print("Waiting 5 seconds before next attempt...")
            time.sleep(5)

def main():
    target_address = '16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v'
    public_key_x_hex = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
    public_key_x = int(public_key_x_hex[2:], 16)
    
    # Elliptic curve parameters
    g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199B0C75643B8F8E4F
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    min_range = 0x4000000000000000000000000000000000
    max_range = 0x7fffffffffffffffffffffffffffffffff

    print("Starting quantum brute-force search...")
    print(f"Target Bitcoin Address: {target_address}")
    print(f"Search Range: {hex(min_range)} to {hex(max_range)}")
    
    private_key = quantum_brute_force(public_key_x, g_x, g_y, p, min_range, max_range)

    if private_key is not None:
        private_key_hex = f"{private_key:064x}"
        print(f"\n{'*'*60}")
        print("SEARCH COMPLETED SUCCESSFULLY!")
        print(f"Private key: {private_key_hex}")
        found_address = private_key_to_compressed_address(private_key_hex)
        print(f"Bitcoin address: {found_address}")
        print(f"{'*'*60}")
    else:
        print("\nSearch completed. No private key found.")

if __name__ == "__main__":
    main()
