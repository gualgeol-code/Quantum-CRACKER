#i realy hope you get me some Donation for the Quantum project_ 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai /////
import cupy as cp  # CUDA-accelerated operations using CuPy
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.controlflow.break_loop import BreakLoopPlaceholder
from qiskit.circuit.library import ZGate, MCXGate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Options
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
    token='45dVrviihk2W_qDTnSHrNQfXlp2uM5-E8Fw5X0md9PsK',
    overwrite=True,
    set_as_default=True,
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/7d8b6f65e3bb4d76ad7af2f598cc70ca:a4a8e2da-5225-42c3-bbaf-34dba6dd020e::"
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
    target_state = random.randint(0x10000000000000000000000000000000, 0x1fffffffffffffffffffffffffffffff)
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

    target_address = '1PXAyUB8ZoH3WD8n5zoAthYjN15yN5CVq5'
    quantum_registers = 125
    num_ancillas = 1
    num_iterations = 1  # Kurangi untuk testing, naikkan nanti

    print("Mencari backend yang tersedia...")
    try:
        backends = service.backends()
        print(f"Backends yang tersedia: {[b.name for b in backends]}")
        
        available_hardware = []
        for backend in backends:
            if not backend.configuration().simulator:
                if backend.configuration().n_qubits >= 125:
                    available_hardware.append(backend)
        
        if available_hardware:
            backend = min(available_hardware, key=lambda b: b.status().pending_jobs)
            print(f"Menggunakan hardware: {backend.name} dengan {backend.configuration().n_qubits} qubit")
        else:
            print("Menggunakan simulator...")
            # Coba beberapa kemungkinan nama simulator
            simulator_names = ["ibmq_qasm_simulator", "simulator_statevector", "simulator_mps", "simulator"]
            backend = None
            for sim_name in simulator_names:
                try:
                    backend = service.backend(sim_name)
                    print(f"Menggunakan simulator: {backend.name}")
                    break
                except:
                    continue
            
            if backend is None:
                # Jika tidak ada simulator spesifik, ambil backend pertama yang tersedia
                if backends:
                    backend = backends[0]
                    print(f"Menggunakan backend yang tersedia: {backend.name}")
                else:
                    raise Exception("Tidak ada backend yang tersedia")
            
    except Exception as e:
        print(f"Error mencari backend: {e}")
        # Coba ambil backend pertama yang tersedia
        try:
            backends = service.backends()
            if backends:
                backend = backends[0]
                print(f"Menggunakan fallback backend: {backend.name}")
            else:
                raise Exception("Tidak ada backend yang tersedia")
        except:
            print("Tidak dapat menemukan backend. Menghentikan program.")
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

        # =========== PERBAIKAN UTAMA: Menggunakan SamplerV2 tanpa Session ===========
        print("\nMenyiapkan SamplerV2 tanpa Session (mode runtime langsung)...")

        try:
            # BUAT OPTIONS DENGAN STRUKTUR V2 YANG BENAR
            options = Options()
            # PERUBAHAN PENTING: Atur shots dengan cara yang benar untuk V2
            options.default_shots = 1024  # Bukan options.execution.shots
            options.optimization_level = 1

            # BUAT SAMPLER DENGAN BACKEND LANGSUNG (TANPA SESSION)
            sampler = SamplerV2(backend=backend, options=options)
            print("SamplerV2 created successfully")

            # Jalankan circuit
            print(f"Running circuit with {options.default_shots} shots...")
            # Format yang benar untuk SamplerV2.run()
            pub = (transpiled_circuit,)
            job = sampler.run([pub])

            job_id = job.job_id()
            print(f"Job submitted successfully. Job ID: {job_id}")

            print("Waiting for job results...")
            job.wait_for_final_state()

            # Dapatkan hasil
            result = job.result()
            print("Job completed successfully")

            # Ekstrak counts dari hasil
            counts = {}
            if hasattr(result[0].data, 'c') and result[0].data.c is not None:
                counts_array = result[0].data.c
                
                for i, count_val in enumerate(counts_array):
                    if count_val > 0:
                        bin_key = format(i, f'0{quantum_registers}b')
                        counts[bin_key] = count_val
                        
                print(f"Got {len(counts)} unique measurement results")
                
                # Cek setiap hasil
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
            
        except Exception as e:
            print(f"Error in quantum execution: {e}")
            print("Trying with other available backends instead...")
            
            # Coba semua backend yang tersedia, ambil yang pertama (biasanya simulator)
            try:
                all_backends = service.backends()
                if all_backends:
                    # Cari backend yang bukan yang sedang digunakan
                    for new_backend in all_backends:
                        if new_backend.name != backend.name:
                            backend = new_backend
                            print(f"Switched to available backend: {backend.name}")
                            # Continue ke loop berikutnya untuk coba lagi dengan backend ini
                            break
                    else:
                        # Jika tidak ada backend lain, gunakan yang pertama
                        backend = all_backends[0]
                        print(f"Using the only available backend: {backend.name}")
                    
                    # Tunggu sebentar sebelum mencoba lagi
                    time.sleep(5)
                    continue
                else:
                    print("No backends available at all.")
                    return None
            except Exception as sim_error:
                print(f"Failed to switch backend: {sim_error}")
                return None
        
        # Jika menggunakan hardware, kurangi frekuensi attempts
        if not backend.configuration().simulator:
            print("Waiting 30 seconds before next attempt...")
            time.sleep(30)
        else:
            print("Waiting 5 seconds before next attempt...")
            time.sleep(5)

def main():
    target_address = '1PXAyUB8ZoH3WD8n5zoAthYjN15yN5CVq5'
    public_key_x_hex = "0233709eb11e0d4439a729f21c2c443dedb727528229713f0065721ba8fa46f00e"
    public_key_x = int(public_key_x_hex[2:], 16)
    
    # Elliptic curve parameters
    g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199B0C75643B8F8E4F
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    min_range = 0x10000000000000000000000000000000
    max_range = 0x1fffffffffffffffffffffffffffffff

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
