#!/usr/init/env python3
# -*- coding: utf-8 -*-

"""
@file ledger.py
@brief Permissioned, Single-Validator, Hash-Chained Ledger for Edge Nodes.
@details Implements a lightweight blockchain ledger for edge-fog architectures:
         - Edge nodes maintain a circular FIFO pending queue and the last block hash.
         - Blocks are dynamically sealed and signed (ECDSA secp256r1 over SHA-256) 
           when a batch is full or upon encountering critical alerts.
         - Full chain verification and history are maintained on the fog/cloud tier.

Requirements:
    pip install cryptography
"""

import hashlib
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)

# Genesis block initial hash constant
GENESIS_HASH = "0" * 64


def sha256_hex(data: bytes) -> str:
    """Computes the SHA-256 hexadecimal digest of input bytes."""
    return hashlib.sha256(data).hexdigest()


def canonical(obj) -> bytes:
    """Produces deterministic JSON encoding ensuring consistent hashing across nodes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ===========================================================================
# Data Structures
# ===========================================================================

@dataclass
class Transaction:
    """Represents a lightweight security event or flow classification record."""
    timestamp: float
    flow_id: str          # e.g., "src_ip:port-dst_ip:port-proto" (no raw payload)
    prediction: str       # Predicted class label from the TCN model
    alert_type: str       # "NORMAL", "ALERT", or "CRITICAL"
    metadata_hash: str    # SHA-256 hash of original flow metadata

    def digest(self) -> str:
        """Returns the cryptographic digest of the transaction."""
        return sha256_hex(canonical(asdict(self)))


@dataclass
class Block:
    """Represents a sealed block in the hash-chained ledger."""
    index: int
    prev_hash: str
    timestamp: float
    tx_hashes: List[str]
    txs: List[dict]
    block_hash: str = ""
    signature: bytes = b""    # 64-byte raw signature concatenation: r (32B) || s (32B)

    def header_bytes(self) -> bytes:
        """Serializes block header attributes for hashing."""
        return canonical({
            "index": self.index,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
            "tx_hashes": self.tx_hashes,
        })

    def to_dict(self) -> dict:
        """Converts the block instance into a dictionary representation."""
        return {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
            "tx_hashes": self.tx_hashes,
            "txs": self.txs,
            "block_hash": self.block_hash,
            "signature": self.signature.hex(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Block":
        """Reconstructs a Block instance from a dictionary."""
        return cls(
            index=d["index"],
            prev_hash=d["prev_hash"],
            timestamp=d["timestamp"],
            tx_hashes=d["tx_hashes"],
            txs=d["txs"],
            block_hash=d["block_hash"],
            signature=bytes.fromhex(d["signature"]),
        )


# ===========================================================================
# Cryptographic Signer & Verifier
# ===========================================================================

class Signer:
    """Manages ECDSA keypairs (secp256r1) for signing edge ledger blocks."""
    
    def __init__(self, private_key: ec.EllipticCurvePrivateKey):
        self._key = private_key

    @classmethod
    def generate(cls) -> "Signer":
        """Generates a new random ECDSA private key on the secp256r1 curve."""
        return cls(ec.generate_private_key(ec.SECP256R1()))

    @classmethod
    def load_pem(cls, path: str) -> "Signer":
        """Loads a private key from a PEM file."""
        with open(path, "rb") as f:
            return cls(serialization.load_pem_private_key(f.read(), password=None))

    def save_pem(self, path: str) -> None:
        """Saves the private key securely to disk in PEM format."""
        pem = self._key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(pem)

    def public_key(self):
        """Returns the corresponding public key."""
        return self._key.public_key()

    def sign_raw(self, data: bytes) -> bytes:
        """Signs data using ECDSA and returns raw 64-byte (r || s) format."""
        der = self._key.sign(data, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def verify_raw(public_key, data: bytes, sig: bytes) -> bool:
    """Verifies a raw 64-byte ECDSA signature against given data and public key."""
    if len(sig) != 64:
        return False
    r = int.from_bytes(sig[:32], "big")
    s = int.from_bytes(sig[32:], "big")
    try:
        public_key.verify(encode_dss_signature(r, s), data, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


# ===========================================================================
# Edge-Side Ledger Management
# ===========================================================================

class EdgeLedger:
    """Lightweight ledger client running on resource-constrained edge nodes."""
    
    def __init__(self, signer: Signer, block_size: int = 16, max_pending: int = 1024):
        self.signer = signer
        self.block_size = block_size
        # Circular FIFO buffer: drops oldest pending transactions if capacity is reached
        self.pending: deque = deque(maxlen=max_pending)
        self.dropped = 0
        self.last_hash = GENESIS_HASH
        self.next_index = 0
        # Outbox queue holding sealed blocks waiting for fog gateway transmission
        self.outbox: deque = deque()

    def add(self, tx: Transaction, critical: bool = False) -> Optional[Block]:
        """Adds a new transaction to the pending queue and seals a block if necessary."""
        if len(self.pending) == self.pending.maxlen:
            self.dropped += 1
        self.pending.append(tx)
        
        # Seal immediately if critical alert is flagged or block size threshold is met
        if critical or len(self.pending) >= self.block_size:
            return self.seal()
        return None

    def seal(self) -> Optional[Block]:
        """Seals pending transactions into a cryptographically signed block."""
        if not self.pending:
            return None
            
        txs = list(self.pending)
        self.pending.clear()

        blk = Block(
            index=self.next_index,
            prev_hash=self.last_hash,
            timestamp=time.time(),
            tx_hashes=[t.digest() for t in txs],
            txs=[asdict(t) for t in txs],
        )
        blk.block_hash = sha256_hex(blk.header_bytes())
        blk.signature = self.signer.sign_raw(bytes.fromhex(blk.block_hash))

        self.last_hash = blk.block_hash
        self.next_index += 1
        self.outbox.append(blk)
        return blk

    def ack(self, index: int) -> None:
        """Removes a confirmed block from the outbox after fog gateway acknowledgment."""
        self.outbox = deque(b for b in self.outbox if b.index != index)


# ===========================================================================
# Fog-Side Chain Verification
# ===========================================================================

def verify_chain(blocks: List[Block], public_key) -> Tuple[bool, str]:
    """Validates the cryptographic integrity, hash chaining, and signatures of the blockchain."""
    prev = GENESIS_HASH
    for i, b in enumerate(blocks):
        if b.index != blocks[0].index + i:
            return False, f"Block {b.index}: Index sequence gap detected."
        if b.prev_hash != prev:
            return False, f"Block {b.index}: Broken previous hash reference."
        if len(b.tx_hashes) != len(b.txs):
            return False, f"Block {b.index}: Transaction count mismatch."
            
        for tx_dict, h in zip(b.txs, b.tx_hashes):
            if sha256_hex(canonical(tx_dict)) != h:
                return False, f"Block {b.index}: Transaction payload tampered."
                
        if sha256_hex(b.header_bytes()) != b.block_hash:
            return False, f"Block {b.index}: Block header tampered."
            
        if not verify_raw(public_key, bytes.fromhex(b.block_hash), b.signature):
            return False, f"Block {b.index}: Invalid cryptographic signature."
            
        prev = b.block_hash
        
    return True, "Chain verification successful."


# ===========================================================================
# Execution Demo & Verification Test
# ===========================================================================

if __name__ == "__main__":
    print("[*] Initializing Edge Ledger Demo...")
    signer = Signer.generate()
    ledger = EdgeLedger(signer, block_size=4)

    # Simulate generating transactions
    for i in range(10):
        tx = Transaction(
            timestamp=time.time(),
            flow_id=f"10.0.0.{i}:1234-10.0.0.1:80-tcp",
            prediction="BENIGN" if i % 3 else "DoS/DDoS",
            alert_type="CRITICAL" if i == 7 else ("ALERT" if i % 3 == 0 else "NORMAL"),
            metadata_hash=sha256_hex(f"flow-{i}".encode()),
        )
        ledger.add(tx, critical=(i == 7))

    ledger.seal()  # Flush remaining pending transactions
    blocks = list(ledger.outbox)
    print(f"[+] Blocks sealed: {len(blocks)}, Dropped transactions: {ledger.dropped}")

    # Verify blockchain integrity
    ok, msg = verify_chain(blocks, signer.public_key())
    print(f"[*] Initial verification result: {ok} ({msg})")

    # Tampering test simulation
    if blocks:
        print("[*] Simulating data tampering attack on block[1]...")
        blocks[1].txs[0]["prediction"] = "BENIGN"
        ok, msg = verify_chain(blocks, signer.public_key())
        print(f"[*] Verification after tampering: {ok} ({msg})")
