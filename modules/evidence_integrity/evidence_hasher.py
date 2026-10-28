"""Cryptographic integrity for extracted video evidence.

Every evidence entry appended to the detection log is hashed with SHA-256,
chained to the hash of the previous entry (Genesis -> H1 -> H2 ...). Any
edit, deletion or reordering of historical log rows breaks the chain, which
makes the exported evidence report tamper-evident. `verify_log` re-computes
the chain from the raw CSV and reports the first row where integrity fails.
"""
import csv
import hashlib
import json
import os
from datetime import datetime

GENESIS_HASH = "0" * 64


class EvidenceHasher:
    """Maintains a SHA-256 hash chain over the evidence log."""

    def __init__(self, chain_file="evidence_chain.json"):
        self.chain_file = chain_file
        self.chain = self._load_chain()

    def _load_chain(self):
        if os.path.exists(self.chain_file):
            with open(self.chain_file, "r") as f:
                return json.load(f)
        return []

    def _save_chain(self):
        with open(self.chain_file, "w") as f:
            json.dump(self.chain, f, indent=2)

    @staticmethod
    def _hash_entry(previous_hash, entry_fields):
        payload = previous_hash + "|" + "|".join(str(v) for v in entry_fields)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def append_entry(self, entry_fields):
        """Hash one evidence entry and append it to the chain.

        `entry_fields` is the ordered list of values written to the CSV row.
        Returns the entry hash so callers can surface it if needed.
        """
        previous_hash = self.chain[-1]["entry_hash"] if self.chain else GENESIS_HASH
        entry_hash = self._hash_entry(previous_hash, entry_fields)
        self.chain.append({
            "sequence": len(self.chain) + 1,
            "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "previous_hash": previous_hash,
            "entry_hash": entry_hash,
        })
        self._save_chain()
        return entry_hash

    def seal_report(self, report_file="evidence_report.json"):
        """Produce the tamper-evident evidence report for the current chain."""
        report = {
            "report_generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "total_entries": len(self.chain),
            "genesis_hash": GENESIS_HASH,
            "head_hash": self.chain[-1]["entry_hash"] if self.chain else GENESIS_HASH,
            "chain": self.chain,
        }
        report_payload = json.dumps(report, sort_keys=True).encode("utf-8")
        report["report_hash"] = hashlib.sha256(report_payload).hexdigest()
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        return report

    def verify_log(self, csv_log="detection_log.csv"):
        """Re-compute the chain from the raw CSV and compare with the stored
        chain. Returns (is_valid, details)."""
        if not os.path.exists(csv_log):
            return False, "evidence log not found"
        previous_hash = GENESIS_HASH
        row_count = 0
        with open(csv_log, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for index, row in enumerate(reader):
                row_count += 1
                expected = self._hash_entry(previous_hash, row)
                if index >= len(self.chain):
                    return False, f"row {index + 1} was added after the chain was sealed"
                stored = self.chain[index]["entry_hash"]
                if stored != expected:
                    return False, f"integrity failure at row {index + 1}: log was modified"
                previous_hash = expected
        if len(self.chain) > row_count:
            return False, f"{len(self.chain) - row_count} entries were deleted from the log"
        return True, f"all {len(self.chain)} entries verified intact"
