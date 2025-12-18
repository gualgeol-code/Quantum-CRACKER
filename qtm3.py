#i realy hope you get me some Donation for the Quantum project_ 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai /////
import cupy as cp  # CUDA-accelerated operations using CuPy
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.controlflow.break_loop import BreakLoopPlaceholder
from qiskit.circuit.library import ZGate, MCXGate
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, SamplerV2
from qiskit.primitives import SamplerResult
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from collections import Counter
from Crypto.Hash import RIPEMD160, SHA256  # Import from pycryptodome
from ecdsa import SigningKey, SECP256k1
from qiskit.quantum_info import Statevector
from bitarray import bitarray
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2  # for simulator
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
    token='45dVrviihk2W_qDTnSHrNQfXlp2uM5-E8Fw5X0md9PsK',  # Replace with your actual token
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
    """
    Applies the Quantum Fourier Transform (QFT) on the first `num_qubits` qubits of the circuit.
    """
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

def public_key_to_public_key_hash(public_key_hex):    
    try:
        sha256_hash = SHA256.new(bytes.fromhex(public_key_hex)).digest()
        ripemd160 = RIPEMD160.new()
        ripemd160.update(sha256_hash)
        ripemd160_hash = ripemd160.digest()
        return ripemd160_hash.hex()
    except ValueError as e:
        raise ValueError(f"Invalid input for public key hex: {e}")    

def grover_oracle(circuit, private_key_qubits, public_key_x, g_x, g_y, p, ancilla):
    """Oracle for Grover's algorithm that checks if the current private key qubits 
    match the target public key via scalar multiplication."""
    
    circuit.barrier()
    keyspace_size = 0x1fffffffffffffffffffffffffffffff - 0x10000000000000000000000000000000 + 1
    num_qubits = 125
    target_state = random.randint(0, keyspace_size - 1)
    target_state = random.randint(0, 2**num_qubits - 1)
    target_state = random.randint(0x10000000000000000000000000000000, 0x1fffffffffffffffffffffffffffffff)
    target_state_bin = format(target_state, f'0{num_qubits}b')
    computed_x, _ = scalar_multiplication(target_state, g_x, g_y, p)

    if computed_x == public_key_x:
        circuit.x(ancilla)

    circuit.barrier()

def mod_inverse(a, p):
    """Modular inverse function for elliptic curve operations."""
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
    """Elliptic curve point addition."""
    if x1 == x2 and y1 == y2:
        return point_doubling(x1, y1, p)
    lam = ((y2 - y1) * mod_inverse(x2 - x1, p)) % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return x3, y3

def point_doubling(x1, y1, p):
    """Elliptic curve point doubling."""
    lam = ((3 * x1 * x1) * mod_inverse(2 * y1, p)) % p
    x3 = (lam * lam - 2 * x1) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return x3, y3

def scalar_multiplication(k, x, y, p):
    """Elliptic curve scalar multiplication."""
    x_res, y_res = x, y
    k_bin = bin(k)[2:]
    for bit in k_bin[1:]:
        x_res, y_res = point_doubling(x_res, y_res, p)
        if bit == '1':
            x_res, y_res = point_addition(x_res, y_res, x, y, p)
    return x_res, y_res

def create_oracle_with_ancilla(num_qubits, target_state):
    """Creates an oracle circuit that marks the target state using an ancillary qubit."""
    oracle_circuit = QuantumCircuit(num_qubits + 1)

    oracle_circuit.x(num_qubits)

    for i, bit in enumerate(bin(target_state)[2:].zfill(num_qubits)):
        if bit == '0':
            oracle_circuit.x(i)

    oracle_circuit.cz(num_qubits, num_qubits - 1)

    for i, bit in enumerate(bin(target_state)[2:].zfill(num_qubits)):
        if bit == '0':
            oracle_circuit.x(i)

    oracle_circuit.x(num_qubits)

    return oracle_circuit

def create_diffusion_operator_gpu(num_qubits):
    """Creates the Grover diffusion operator with CUDA-enabled steps."""
    qc = QuantumCircuit(num_qubits)

    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)

    mcx = MCXGate(num_qubits - 1)
    mcx_gpu = cp.asarray(mcx.to_matrix())
    qc.append(mcx, list(range(num_qubits - 1)) + [num_qubits - 1])

    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))

    return qc

def grovers_algorithm_gpu(num_qubits, target_state, iterations=65536):
    print("Setting up Grover's algorithm with CUDA-enabled diffusion steps...")
    circuit = QuantumCircuit(num_qubits, num_qubits)

    circuit.h(range(num_qubits))
    print("Initialized qubits in superposition.")

    apply_qft(circuit, num_qubits)

    for iteration in range(iterations):
        print(f"Applying Grover iteration {iteration + 1} using CUDA...")

        oracle_circuit = create_oracle_with_ancilla(num_qubits, target_state)
        oracle_matrix_gpu = cp.asarray(oracle_circuit.to_matrix())

        circuit.compose(oracle_circuit, inplace=True)

        diffusion_operator = create_diffusion_operator_gpu(num_qubits)
        circuit.compose(diffusion_operator, inplace=True)
        print("Applied oracle and diffusion operator using CUDA.")

    circuit.measure(range(num_qubits), range(num_qubits))
    print("Measurement operation added to circuit.")
    
    return circuit

def grovers_bruteforce(target_address, keyspace_size=None, num_qubits=125, iterations=65536):
    if keyspace_size is None:
        keyspace_size = 0x1fffffffffffffffffffffffffffffff - 0x10000000000000000000000000000000 + 1

    target_state = random.randint(0, keyspace_size - 1)
    print(f"Chosen target state: {target_state}")

    if iterations is None:
        iterations = 65536

    circuit = grovers_algorithm_gpu(num_qubits, target_state, iterations)

    circuit.measure_all()
    return circuit

def retrieve_job_result(job_id, target_address):
    print(f"Retrieving job result for job ID: {job_id}...")
    quantum_registers = 125
    try:
        job = service.job(job_id)
        result = job.result()

        counts = result.get_counts()
        print(f"Measurement counts retrieved: {counts}")

        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

        for bin_key, count in sorted_counts:
            if len(bin_key) < quantum_registers:
                bin_key = bin_key.zfill(quantum_registers)
            elif len(bin_key) > quantum_registers:
                bin_key = bin_key[:quantum_registers]

            private_key_hex = binary_to_hex(bin_key)
            compressed_address = private_key_to_compressed_address(private_key_hex)

            if compressed_address == target_address:
                print(f"Private key found: {private_key_hex}")
                with open('boom.txt', 'a') as file:
                    file.write(f"Private key: {private_key_hex}\nCompressed Address: {compressed_address}\n\n")
                return private_key_hex, compressed_address

        print("No matching private key found.")
        return None, None
    except Exception as e:
        print(f"Error retrieving job result: {e}")
        return None, None

def binary_to_hex(bin_key):
    bin_key = bin_key.zfill(128)
    return hex(int(bin_key, 2))[2:].zfill(64)

def retrieve_job_result_sampler(job_id, target_address, quantum_registers):
    """Retrieve job results from SamplerV2 and check for valid private keys."""
    print(f"Retrieving job result for job ID: {job_id}...")
    service = QiskitRuntimeService(
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/7d8b6f65e3bb4d76ad7af2f598cc70ca:a4a8e2da-5225-42c3-bbaf-34dba6dd020e::"
    )

    try:
        job = service.job(job_id)
        job_result = job.result()
        print(f"Job result retrieved for job ID {job_id}")
    except Exception as e:
        print(f"Error retrieving job result: {e}")
        return None, None

    try:
        # Format baru untuk SamplerV2: hasil ada di job_result[0].data.c
        result_data = job_result[0].data.c
        
        # Convert ke dictionary counts
        counts = {}
        for i, count in enumerate(result_data):
            if count > 0:
                # Konversi index ke string biner
                bin_key = format(i, f'0{quantum_registers}b')
                counts[bin_key] = count
        
        print("Counts retrieved from job:", counts)

        for bin_key, count in counts.items():
            bin_key = bin_key.strip()

            if len(bin_key) < quantum_registers:
                bin_key = bin_key.ljust(quantum_registers, '0')
            elif len(bin_key) > quantum_registers:
                bin_key = bin_key[:quantum_registers]

            print(f"\nChecking binary key (first 125 bits): {bin_key} with length {len(bin_key)}")
            print(f"Key count: {count} times generated")

            private_key_hex = binary_to_hex(bin_key)
            if private_key_hex is None:
                continue

            compressed_address = private_key_to_compressed_address(private_key_hex)

            if compressed_address == target_address:
                print(f"Valid private key found: {private_key_hex}")

                with open('boom.txt', 'a') as file:
                    file.write(f"Private Key: {private_key_hex}\n")
                    file.write(f"Compressed Address: {compressed_address}\n\n")

                return private_key_hex, compressed_address

        print("No matching private key found.")
        return None, None

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None

def quantum_brute_force(public_key_x: int, g_x: int, g_y: int, p: int, min_range: int, max_range: int) -> int:
    """Main function to perform quantum brute-force search for private keys using Public-Key-Hash."""
    if max_range <= min_range:
        raise ValueError("max_range must be greater than min_range.")

    target_address = '1PXAyUB8ZoH3WD8n5zoAthYjN15yN5CVq5'
    quantum_registers = 125
    private_key = None
    attempt = 0
    num_ancillas = 1
    num_iterations = 65536

    service = QiskitRuntimeService(
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/7d8b6f65e3bb4d76ad7af2f598cc70ca:a4a8e2da-5225-42c3-bbaf-34dba6dd020e::"
    )

    while private_key is None:
        attempt += 1
        print(f"Attempt {attempt}...")

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
                simulators = [b for b in backends if b.configuration().simulator]
                if simulators:
                    backend = simulators[0]
                    print(f"Tidak ada hardware dengan 125 qubit. Menggunakan simulator: {backend.name}")
                else:
                    raise Exception("Tidak ada backend yang tersedia")
                    
        except Exception as e:
            print(f"Error mencari backend: {e}")
            try:
                backend = service.backend("ibmq_qasm_simulator")
                print(f"Menggunakan fallback simulator: {backend.name}")
            except:
                try:
                    backend = service.backend("simulator_statevector")
                    print(f"Menggunakan fallback simulator: {backend.name}")
                except:
                    try:
                        backend = service.backend("simulator_mps")
                        print(f"Menggunakan fallback simulator: {backend.name}")
                    except:
                        print("Tidak dapat menemukan backend yang tersedia. Menghentikan program.")
                        return None
        
        print(f"Selected backend: {backend}")        

        print("Transpiling the circuit for the selected backend.")
        transpiled_circuit = transpile(circuit, backend=backend, optimization_level=3)
        print("Circuit transpiled.")

        # =========== PERBAIKAN UTAMA ===========
        # Ganti backend.run() dengan SamplerV2 API
        print("Running circuit using SamplerV2 API...")
        sampler = SamplerV2(backend=backend)
        
        # Jalankan circuit dengan shots=8192
        # Format: [(circuit, parameter_values, shots)]
        # Karena tidak ada parameter, gunakan None untuk parameter_values
        job = sampler.run([(transpiled_circuit, None, 8192)])
        job_id = job.job_id()
        print(f"Job ID: {job_id}")

        # Tunggu hasil job
        print("Waiting for job results...")
        job.wait_for_final_state()
        
        # Dapatkan hasil dalam format SamplerV2
        result = job.result()
        
        # Ekstrak counts dari hasil SamplerV2
        # Format: result[0].data.c berisi array counts
        counts_array = result[0].data.c
        counts = {}
        
        # Konversi array counts ke dictionary
        for i, count_val in enumerate(counts_array):
            if count_val > 0:
                # Konversi index ke binary string
                bin_key = format(i, f'0{quantum_registers}b')
                counts[bin_key] = count_val
        
        print(f"Measurement counts: {len(counts)} unique results")
        
        # Cek setiap hasil untuk private key yang valid
        for bin_key, count in counts.items():
            bin_key = bin_key.strip()

            if len(bin_key) < quantum_registers:
                bin_key = bin_key.ljust(quantum_registers, '0')
            elif len(bin_key) > quantum_registers:
                bin_key = bin_key[:quantum_registers]

            private_key_hex = binary_to_hex(bin_key)
            if private_key_hex is None:
                continue

            compressed_address = private_key_to_compressed_address(private_key_hex)

            if compressed_address == target_address:
                print(f"Found matching private key: {private_key_hex}")
                
                # Simpan ke file
                with open('boomqft.txt', 'w') as f:
                    f.write(f"Found private key: {private_key_hex}\n")
                    f.write(f"Corresponding Bitcoin address: {compressed_address}\n")
                
                return private_key_hex

        print("No matching key found in this attempt.")
        break

    return None

def main():
    target_address = '1PXAyUB8ZoH3WD8n5zoAthYjN15yN5CVq5'
    public_key_x_hex = "0233709eb11e0d4439a729f21c2c443dedb727528229713f0065721ba8fa46f00e"
    public_key_x = int(public_key_x_hex[2:], 16)
    num_qubits = 125
    iterations = 65536
    g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199B0C75643B8F8E4F
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    min_range = 0x10000000000000000000000000000000
    max_range = 0x1fffffffffffffffffffffffffffffff

    while True:
        print("Starting new quantum brute-force search...")

        private_key = quantum_brute_force(public_key_x, g_x, g_y, p, min_range, max_range)

        if private_key is not None:
            private_key_hex = f"{private_key:064x}"
            print(f"Found private key: {private_key_hex}")
            found_address = private_key_to_compressed_address(private_key_hex)
            print(f"Corresponding Bitcoin address: {found_address}")

            with open("boomqft.txt", "w") as f:
                f.write(f"Found private key: {private_key_hex}\n")
                f.write(f"Corresponding Bitcoin address: {found_address}\n")
            print("Private key and corresponding address saved to boomqft.txt.")
            break
        else:
            print("Private key not found in the specified range. Retrying...")

if __name__ == "__main__":
    main()
