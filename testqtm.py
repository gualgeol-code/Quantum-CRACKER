import numpy as np
import hashlib
import base58
import time
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_aer import AerSimulator
from Crypto.Hash import RIPEMD160, SHA256
from ecdsa import SigningKey, SECP256k1

# --- KONFIGURASI AKUN ---
# Pastikan token Anda valid untuk mengakses paket 'open'
try:
    service = QiskitRuntimeService(
        channel='ibm_quantum_platform', 
        token='45dVrviihk2W_qDTnSHrNQfXlp2uM5-E8Fw5X0md9PsK',
        overwrite=True
    )
except Exception as e:
    print(f"Peringatan: Gagal login ke IBM. Menggunakan simulator lokal. Error: {e}")
    service = None

def get_best_backend(num_qubits):
    """Mencari backend yang tersedia secara dinamis untuk paket free."""
    if service:
        try:
            # Mencari hardware asli yang paling tidak sibuk dengan qubit yang cukup
            backend = service.least_busy(min_qubits=num_qubits, simulator=False, operational=True)
            print(f"Hardware terpilih: {backend.name}")
            return backend
        except:
            print("Tidak ada hardware free tersedia. Menggunakan Simulator Cloud.")
            return service.backend("ibmq_qasm_simulator")
    return AerSimulator()

# --- OPERATOR KUANTUM ---

def grover_oracle(circuit, qubits, target_int, ancilla):
    """
    Oracle Kuantum: Membalikkan fase state yang cocok dengan target_int.
    Ini adalah perbaikan utama agar pengecekan terjadi di sirkuit kuantum.
    """
    num_qubits = len(qubits)
    # Konversi target ke biner dalam urutan bit sirkuit
    target_bin = format(target_int, f'0{num_qubits}b')[::-1] 

    # 1. Komputasi: Gunakan gerbang X untuk mengubah target '0' menjadi '1' 
    # agar bisa dideteksi oleh Multi-Controlled X (MCX)
    for i in range(num_qubits):
        if target_bin[i] == '0':
            circuit.x(qubits[i])

    # 2. Marking: Gunakan MCX pada ancilla untuk Phase Kickback
    circuit.mcx(qubits, ancilla)

    # 3. Uncompute: Kembalikan bit X ke posisi semula
    for i in range(num_qubits):
        if target_bin[i] == '0':
            circuit.x(qubits[i])

def grover_diffusion(circuit, qubits, ancilla):
    """Operator Difusi: Inversi di sekitar nilai rata-rata."""
    num_qubits = len(qubits)
    circuit.h(qubits)
    circuit.x(qubits)
    
    # Gerbang Multi-Controlled Z menggunakan ancilla
    circuit.h(qubits[-1])
    circuit.mcx(qubits[:-1], qubits[-1])
    circuit.h(qubits[-1])
    
    circuit.x(qubits)
    circuit.h(qubits)

# --- FUNGSI UTILITAS BITCOIN ---

def private_key_to_address(pk_int):
    """Konversi integer private key ke Bitcoin Address (Compressed)."""
    pk_hex = hex(pk_int)[2:].zfill(64)
    pk_bytes = bytes.fromhex(pk_hex)
    sk = SigningKey.from_string(pk_bytes, curve=SECP256k1)
    vk = sk.verifying_key
    
    # Compressed Public Key
    prefix = b'\x02' if vk.to_string()[63] % 2 == 0 else b'\x03'
    public_key = prefix + vk.to_string()[:32]
    
    # Hashing
    sha = hashlib.sha256(public_key).digest()
    rip = RIPEMD160.new(sha).digest()
    ver = b'\x00' + rip
    chk = hashlib.sha256(hashlib.sha256(ver).digest()).digest()[:4]
    return base58.b58encode(ver + chk).decode('utf-8')

# --- MAIN PROGRAM ---

def main():
    # Catatan: Kita gunakan 10-bit untuk demo teknis. 
    # 125-bit membutuhkan hardware masa depan (Fault-Tolerant).
    num_qubits = 10 
    target_secret = 742 # Kunci yang ceritanya kita cari
    target_address = '1PXAyUB8ZoH3WD8n5zoAthYjN15yN5CVq5'

    qr = QuantumRegister(num_qubits, 'q')
    an = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(num_qubits, 'c')
    qc = QuantumCircuit(qr, an, cr)

    # Inisialisasi: Superposisi & Ancilla dalam state |->
    qc.h(qr)
    qc.x(an)
    qc.h(an)
    qc.barrier()

    # Iterasi Grover (1-2 iterasi untuk hardware NISQ agar noise tidak terlalu tinggi)
    iterations = 1
    for _ in range(iterations):
        grover_oracle(qc, qr, target_secret, an)
        qc.barrier()
        grover_diffusion(qc, qr, an)
        qc.barrier()

    # Pengukuran
    qc.measure(qr, cr)

    # Eksekusi
    backend = get_best_backend(num_qubits + 1)
    transpiled_qc = transpile(qc, backend, optimization_level=3)
    
    print(f"Menjalankan sirkuit... Kedalaman: {transpiled_qc.depth()}")
    
    try:
        if hasattr(backend, 'run'): # Local Aer
            job = backend.run(transpiled_qc, shots=1024)
        else: # Runtime Cloud
            sampler = SamplerV2(backend)
            job = sampler.run([transpiled_qc])
        
        result = job.result()
        # Ambil counts (mendukung struktur hasil V2)
        if hasattr(result, 'get_counts'):
            counts = result.get_counts()
        else:
            counts = result[0].data.c.get_counts()
        
        # Analisis hasil
        top_binary = max(counts, key=counts.get)
        found_int = int(top_binary, 2)
        
        print(f"\nHasil Pengukuran Tertinggi: {top_binary}")
        print(f"Kunci Ditemukan (Dec): {found_int}")

        if found_int == target_secret:
            print("STATUS: SUKSES! Kunci ditemukan.")
            final_address = private_key_to_address(found_int)
            with open("boom.txt", "a") as f:
                f.write(f"Private Key (Dec): {found_int}\nAddress: {final_address}\n\n")
        else:
            print("STATUS: Belum cocok. Cobalah tingkatkan iterasi atau gunakan simulator.")

    except Exception as e:
        print(f"Gagal menjalankan pekerjaan: {e}")

if __name__ == "__main__":
    main()
