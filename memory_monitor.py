#!/usr/bin/env python3
"""
Memory monitoring utility for debugging memory usage in the Flask app.
"""

import psutil
import os
import time
import gc

def get_memory_info():
    """Get current memory usage information."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': round(memory_info.rss / 1024 / 1024, 2),  # Resident Set Size
        'vms_mb': round(memory_info.vms / 1024 / 1024, 2),  # Virtual Memory Size
        'percent': round(process.memory_percent(), 2),
        'available_mb': round(psutil.virtual_memory().available / 1024 / 1024, 2),
        'total_mb': round(psutil.virtual_memory().total / 1024 / 1024, 2)
    }

def log_memory_usage(context=""):
    """Log current memory usage with context."""
    info = get_memory_info()
    print(f"[MEMORY] {context}: RSS={info['rss_mb']}MB, VMS={info['vms_mb']}MB, "
          f"Usage={info['percent']}%, Available={info['available_mb']}MB")
    return info

def force_garbage_collection():
    """Force garbage collection and log memory before/after."""
    before = get_memory_info()
    gc.collect()
    after = get_memory_info()
    
    freed = before['rss_mb'] - after['rss_mb']
    print(f"[GC] Freed {freed:.2f}MB (Before: {before['rss_mb']}MB, After: {after['rss_mb']}MB)")
    return freed

if __name__ == "__main__":
    # Test the memory monitoring
    log_memory_usage("Startup")
    time.sleep(1)
    force_garbage_collection()
    log_memory_usage("After GC")
