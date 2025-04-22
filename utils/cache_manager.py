import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import hashlib
import json
import sys

class CacheManager:
    def __init__(self):
        if 'cache_data' not in st.session_state:
            st.session_state.cache_data = {}
        if 'cache_timestamps' not in st.session_state:
            st.session_state.cache_timestamps = {}
        # Add memory limit
        self.max_cache_size = 100 * 1024 * 1024  # 100MB
    
    def _get_cache_key(self, key, params=None):
        """Generate a unique cache key"""
        if params:
            key = f"{key}_{hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()}"
        return key
    
    def get(self, key, params=None, ttl=3600):
        """Get data from cache if valid"""
        cache_key = self._get_cache_key(key, params)
        if cache_key in st.session_state.cache_data:
            timestamp = st.session_state.cache_timestamps.get(cache_key)
            if timestamp and (datetime.now() - timestamp).total_seconds() < ttl:
                return st.session_state.cache_data[cache_key]
        return None
    
    def set(self, key, data, params=None):
        """Store data in cache with size check"""
        try:
            # Estimate data size
            data_size = sys.getsizeof(data)
            current_size = sum(sys.getsizeof(v) for v in st.session_state.cache_data.values())
            
            if current_size + data_size > self.max_cache_size:
                # Remove oldest entries
                oldest_key = min(st.session_state.cache_timestamps.items(), key=lambda x: x[1])[0]
                self.clear(oldest_key)
            
            cache_key = self._get_cache_key(key, params)
            st.session_state.cache_data[cache_key] = data
            st.session_state.cache_timestamps[cache_key] = datetime.now()
        except Exception as e:
            st.warning(f"Cache operation failed: {str(e)}")
    
    def clear(self, key=None):
        """Clear specific or all cache entries"""
        if key:
            cache_key = self._get_cache_key(key)
            if cache_key in st.session_state.cache_data:
                del st.session_state.cache_data[cache_key]
                del st.session_state.cache_timestamps[cache_key]
        else:
            st.session_state.cache_data = {}
            st.session_state.cache_timestamps = {}
