"""
Speaker Classification Streamlit App
Allows manual assignment of speakers to diarization labels
With automatic voice matching for same-speaker segments
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import numpy as np
import uuid
import io
from snowflake.snowpark.context import get_active_session

st.set_page_config(page_title="Speaker Classification", page_icon="🎤", layout="wide")

# Get Snowpark session (works in Streamlit-in-Snowflake)
session = get_active_session()

# Use dedicated warehouse for app queries (separate from background tasks)
try:
    session.sql("USE WAREHOUSE STREAMLIT_APP_WH").collect()
except:
    pass  # Fall back to default warehouse

# Configuration
SIMILARITY_THRESHOLD = 0.75  # Threshold for auto-matching

def clear_all_caches():
    """Clear all cached data - compatible with older Streamlit versions"""
    try:
        st.cache_data.clear()
    except:
        pass

# =============================================================================
# CACHED QUERY FUNCTIONS - Use caching to avoid repeated queries
# =============================================================================

import time as _time

def _sql_with_retry(sql, max_retries=3, delay=1):
    """Execute SQL with auto-retry on warehouse resume - for use in cached functions"""
    for attempt in range(max_retries):
        try:
            return session.sql(sql).collect()
        except Exception as e:
            error_str = str(e).lower()
            if attempt < max_retries - 1 and any(x in error_str for x in ['timeout', 'warehouse', 'session', 'connection', 'suspended']):
                _time.sleep(delay * (attempt + 1))
                continue
            raise e
    return []

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_speakers_cached():
    """Get all registered speakers, sorted by meeting count descending - CACHED"""
    results = _sql_with_retry("""
        SELECT speaker_id, display_name, email, COALESCE(meeting_count, 0) as meeting_count 
        FROM SPEAKERS 
        ORDER BY meeting_count DESC, display_name ASC
    """)
    return [(r[0], r[1], r[2], r[3]) for r in results]

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_calls_cached(only_incomplete=True):
    """Get calls - CACHED"""
    filter_clause = "WHERE classification_status != 'completed'" if only_incomplete else ""
    results = _sql_with_retry(f"""
        SELECT m.call_id, m.title, m.call_date, m.classification_status,
               c.total_contributions, c.speaker_count
        FROM CALLS m
        LEFT JOIN (
            SELECT call_id, 
                   COUNT(*) as total_contributions,
                   COUNT(DISTINCT diarization_label) as speaker_count
            FROM CALL_CONTRIBUTIONS
            GROUP BY call_id
        ) c ON m.call_id = c.call_id
        {filter_clause}
        ORDER BY m.title ASC
    """)
    return [(r[0], r[1], r[2], r[3], r[4] or 0, r[5] or 0) for r in results]

@st.cache_data(ttl=30)
def get_diarization_groups_cached(call_id):
    """Get contributions grouped by diarization label - CACHED"""
    try:
        results = _sql_with_retry(f"""
            SELECT 
                diarization_label,
                COUNT(*) as segment_count,
                SUM(end_time_seconds - start_time_seconds) as total_duration,
                MIN(identified_speaker_id) as current_speaker_id
            FROM CALL_CONTRIBUTIONS
            WHERE call_id = '{call_id}'
            GROUP BY diarization_label
            ORDER BY MIN(segment_number)
        """)
        return [(r[0], r[1], r[2], None, r[3]) for r in results]
    except Exception as e:
        st.error(f"Error loading groups: {e}")
        return []

@st.cache_data(ttl=60)
def get_embedding_counts_cached():
    """Get embedding counts - CACHED"""
    try:
        result = _sql_with_retry("""
            SELECT 
                (SELECT COUNT(*) FROM CONTRIBUTION_EMBEDDINGS) as stored,
                (SELECT COUNT(*) FROM CALL_CONTRIBUTIONS 
                 WHERE identified_speaker_id IS NULL 
                 AND duration_seconds >= 5.0
                 AND contribution_id NOT IN (SELECT contribution_id FROM CONTRIBUTION_EMBEDDINGS)) as pending
        """)
        return result[0][0] or 0, result[0][1] or 0
    except:
        return 0, 0

@st.cache_data(ttl=300)  # Cache audio URL for 5 minutes
def get_audio_url_cached(call_id):
    """Get presigned URL for call audio - CACHED"""
    try:
        result = _sql_with_retry(f"SELECT recording_path FROM CALLS WHERE call_id = '{call_id}'")
        if not result or not result[0][0]:
            return None
        recording_path = result[0][0]
        if recording_path.startswith('@'):
            recording_path = recording_path[1:]
        parts = recording_path.split('/')
        file_path = '/'.join(parts[1:]) if len(parts) > 1 else parts[0]
        
        url_result = _sql_with_retry(f"SELECT GET_PRESIGNED_URL(@{parts[0]}, '{file_path}', 3600) as url")
        return url_result[0][0] if url_result else None
    except:
        return None

# =============================================================================
# NON-CACHED FUNCTIONS - For data that changes frequently or needs fresh data
# =============================================================================

def run_query_with_retry(sql, max_retries=3, delay=1):
    """Execute SQL with auto-retry on warehouse resume"""
    for attempt in range(max_retries):
        try:
            return session.sql(sql).collect()
        except Exception as e:
            error_str = str(e).lower()
            if attempt < max_retries - 1 and any(x in error_str for x in ['timeout', 'warehouse', 'session', 'connection', 'suspended']):
                _time.sleep(delay * (attempt + 1))
                continue
            raise e
    return []

def run_query(sql):
    """Execute SQL and return results as list of tuples"""
    return run_query_with_retry(sql)

def get_segments_for_label(call_id, diarization_label):
    """Get individual segments for a diarization label, ordered by time"""
    results = _sql_with_retry(f"""
        SELECT 
            contribution_id,
            segment_number,
            start_time_seconds,
            end_time_seconds,
            text_content
        FROM CALL_CONTRIBUTIONS
        WHERE call_id = '{call_id}'
        AND diarization_label = '{diarization_label}'
        ORDER BY start_time_seconds
    """)
    return [(r[0], r[1], r[2], r[3], r[4]) for r in results]

def increment_speaker_meeting_count(speaker_id):
    """Increment the meeting count for a speaker"""
    try:
        session.sql(f"""
            UPDATE SPEAKERS 
            SET meeting_count = COALESCE(meeting_count, 0) + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE speaker_id = '{speaker_id}'
        """).collect()
        return True
    except:
        return False

def assign_speaker_fast(call_id, diarization_label, speaker_id):
    """
    Fast speaker assignment - minimal database operations only.
    """
    try:
        # Single UPDATE statement - that's all we need for immediate feedback
        session.sql(f"""
            UPDATE CALL_CONTRIBUTIONS
            SET identified_speaker_id = '{speaker_id}',
                identification_method = 'manual',
                identification_confidence = 1.0,
                classification_status = 'classified'
            WHERE call_id = '{call_id}' 
            AND diarization_label = '{diarization_label}'
        """).collect()
        
        return {'success': True, 'segments_updated': 1, 'errors': []}
        
    except Exception as e:
        return {'success': False, 'segments_updated': 0, 'errors': [str(e)]}

def delete_contributions(call_id, diarization_label):
    """Delete all contributions for a specific diarization label"""
    try:
        count_result = run_query(f"""
            SELECT COUNT(*) FROM CALL_CONTRIBUTIONS
            WHERE call_id = '{call_id}' 
            AND diarization_label = '{diarization_label}'
        """)
        count = count_result[0][0] if count_result else 0
        
        if count == 0:
            return False, "No contributions found"
        
        session.sql(f"""
            DELETE FROM CLASSIFICATION_QUEUE 
            WHERE call_id = '{call_id}'
            AND diarization_label = '{diarization_label}'
        """).collect()
        
        session.sql(f"""
            DELETE FROM CALL_CONTRIBUTIONS
            WHERE call_id = '{call_id}' 
            AND diarization_label = '{diarization_label}'
        """).collect()
        
        clear_all_caches()
        return True, f"{count} contributions deleted"
    except Exception as e:
        return False, str(e)

def create_speaker(name, email=None, department=None, company=None, notes=None, meeting_count=0):
    """Create a new speaker"""
    speaker_id = f"SPK_{uuid.uuid4().hex[:16]}"
    
    try:
        name_escaped = name.replace("'", "''")
        email_val = f"'{email}'" if email else 'NULL'
        dept_val = f"'{department}'" if department else 'NULL'
        company_val = f"'{company}'" if company else 'NULL'
        notes_val = f"'{notes}'" if notes else 'NULL'
        
        session.sql(f"""
            INSERT INTO SPEAKERS (speaker_id, display_name, email, department, company, notes, is_internal, meeting_count, created_at, updated_at, created_by)
            VALUES ('{speaker_id}', '{name_escaped}', {email_val}, {dept_val}, {company_val}, {notes_val}, TRUE, {meeting_count}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_USER())
        """).collect()
        clear_all_caches()
        return speaker_id
    except Exception as e:
        st.error(f"Error creating speaker: {e}")
        return None

def get_speaker_details(speaker_id):
    """Get full details of a speaker"""
    results = run_query(f"""
        SELECT speaker_id, display_name, email, department, company, notes, is_internal, COALESCE(meeting_count, 0)
        FROM SPEAKERS
        WHERE speaker_id = '{speaker_id}'
    """)
    if results:
        r = results[0]
        return (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7])
    return None

def update_speaker(speaker_id, name, email=None, department=None, company=None, notes=None, is_internal=True, meeting_count=None):
    """Update an existing speaker"""
    try:
        name_escaped = name.replace("'", "''")
        email_val = f"'{email}'" if email else 'NULL'
        dept_val = f"'{department}'" if department else 'NULL'
        company_val = f"'{company}'" if company else 'NULL'
        notes_escaped = notes.replace("'", "''") if notes else None
        notes_val = f"'{notes_escaped}'" if notes_escaped else 'NULL'
        internal_val = 'TRUE' if is_internal else 'FALSE'
        meeting_count_sql = f", meeting_count = {meeting_count}" if meeting_count is not None else ""
        
        session.sql(f"""
            UPDATE SPEAKERS
            SET display_name = '{name_escaped}',
                email = {email_val},
                department = {dept_val},
                company = {company_val},
                notes = {notes_val},
                is_internal = {internal_val},
                updated_at = CURRENT_TIMESTAMP
                {meeting_count_sql}
            WHERE speaker_id = '{speaker_id}'
        """).collect()
        clear_all_caches()
        return True
    except Exception as e:
        st.error(f"Error updating speaker: {e}")
        return False

def get_speaker_details(speaker_id):
    """Get full details of a speaker"""
    try:
        results = run_query(f"""
            SELECT speaker_id, display_name, email, department, company, notes, is_internal, meeting_count
            FROM SPEAKERS
            WHERE speaker_id = '{speaker_id}'
        """)
        if results:
            r = results[0]
            return {
                'speaker_id': r[0],
                'display_name': r[1],
                'email': r[2],
                'department': r[3],
                'company': r[4],
                'notes': r[5],
                'is_internal': r[6],
                'meeting_count': r[7]
            }
        return None
    except:
        return None

def delete_speaker(speaker_id):
    """Delete a speaker (only if not used)"""
    try:
        count_result = run_query(f"""
            SELECT COUNT(*) FROM CALL_CONTRIBUTIONS
            WHERE identified_speaker_id = '{speaker_id}'
        """)
        count = count_result[0][0] if count_result else 0
        
        if count > 0:
            return False, f"Speaker is used in {count} contributions"
        
        session.sql(f"DELETE FROM SPEAKERS WHERE speaker_id = '{speaker_id}'").collect()
        clear_all_caches()
        return True, None
    except Exception as e:
        return False, str(e)

def update_call_status(call_id):
    """Update call status based on classification progress.
    
    A call is 'completed' only when ALL diarization labels have an assigned speaker.
    This prevents marking as complete when some labels are deleted but others remain unassigned.
    """
    # Check if all diarization labels have a speaker assigned
    results = run_query(f"""
        SELECT 
            COUNT(DISTINCT diarization_label) as total_labels,
            COUNT(DISTINCT CASE WHEN identified_speaker_id IS NOT NULL THEN diarization_label END) as assigned_labels
        FROM CALL_CONTRIBUTIONS
        WHERE call_id = '{call_id}'
    """)
    
    if results:
        total_labels, assigned_labels = results[0][0], results[0][1]
        if total_labels > 0 and total_labels == assigned_labels:
            session.sql(f"""
                UPDATE CALLS SET classification_status = 'completed'
                WHERE call_id = '{call_id}'
            """).collect()
            clear_all_caches()
        else:
            # Reset to pending if not all labels are assigned
            session.sql(f"""
                UPDATE CALLS SET classification_status = 'pending'
                WHERE call_id = '{call_id}' AND classification_status = 'completed'
            """).collect()
            clear_all_caches()

def update_call_title(call_id, new_title):
    """Update the title of a call"""
    try:
        title_escaped = new_title.replace("'", "''")
        session.sql(f"""
            UPDATE CALLS SET title = '{title_escaped}'
            WHERE call_id = '{call_id}'
        """).collect()
        clear_all_caches()
        return True, None
    except Exception as e:
        return False, str(e)

def delete_call(call_id):
    """
    Delete a call and all its related data.
    Does NOT delete speaker voiceprints (shared across meetings).
    """
    try:
        # Get recording path before deleting
        result = run_query(f"SELECT recording_path FROM CALLS WHERE call_id = '{call_id}'")
        recording_path = result[0][0] if result and result[0][0] else None
        
        # Delete from CONTRIBUTION_EMBEDDINGS (for this call's contributions only)
        session.sql(f"""
            DELETE FROM CONTRIBUTION_EMBEDDINGS
            WHERE contribution_id IN (
                SELECT contribution_id FROM CALL_CONTRIBUTIONS
                WHERE call_id = '{call_id}'
            )
        """).collect()
        
        # Delete from CLASSIFICATION_QUEUE
        session.sql(f"""
            DELETE FROM CLASSIFICATION_QUEUE 
            WHERE call_id = '{call_id}'
        """).collect()
        
        # Delete from CALL_CONTRIBUTIONS
        session.sql(f"""
            DELETE FROM CALL_CONTRIBUTIONS
            WHERE call_id = '{call_id}'
        """).collect()
        
        # Delete the MP3 file from stage
        if recording_path:
            try:
                # recording_path format: @CALL_RECORDINGS/filename.mp3
                path = recording_path.lstrip('@')
                if '/' in path:
                    parts = path.split('/', 1)
                    stage_name = parts[0]
                    file_path = parts[1]
                    session.sql(f"REMOVE @{stage_name}/{file_path}").collect()
            except Exception as e:
                # Don't fail if file removal fails
                pass
        
        # Delete from CALLS
        session.sql(f"""
            DELETE FROM CALLS
            WHERE call_id = '{call_id}'
        """).collect()
        
        clear_all_caches()
        return True, None
    except Exception as e:
        return False, str(e)

def filter_speakers(speakers, search_term):
    """Filter speakers by search term"""
    if not search_term:
        return speakers
    search_lower = search_term.lower()
    return [s for s in speakers if search_lower in (s[1] or '').lower() or search_lower in (s[2] or '').lower()]

def import_speakers_from_csv(csv_content):
    """Import speakers from CSV content"""
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        df.columns = df.columns.str.lower().str.strip()
        
        if 'name' not in df.columns and 'display_name' not in df.columns:
            return False, "CSV must have a 'name' or 'display_name' column"
        
        name_col = 'name' if 'name' in df.columns else 'display_name'
        imported = 0
        updated = 0
        
        for idx, row in df.iterrows():
            name = str(row[name_col]).strip() if pd.notna(row[name_col]) else None
            if not name:
                continue
            
            email = str(row.get('email', '')).strip() if pd.notna(row.get('email')) else None
            department = str(row.get('department', '')).strip() if pd.notna(row.get('department')) else None
            company = str(row.get('company', '')).strip() if pd.notna(row.get('company')) else None
            meeting_count = int(row.get('meeting_count', 0)) if pd.notna(row.get('meeting_count')) else 0
            
            existing = None
            if email:
                existing = run_query(f"SELECT speaker_id FROM SPEAKERS WHERE email = '{email}'")
            if not existing:
                name_escaped = name.replace("'", "''")
                existing = run_query(f"SELECT speaker_id FROM SPEAKERS WHERE display_name = '{name_escaped}'")
            
            try:
                if existing:
                    speaker_id = existing[0][0]
                    update_speaker(speaker_id, name, email, department, company, None, True, meeting_count)
                    updated += 1
                else:
                    create_speaker(name, email, department, company, None, meeting_count)
                    imported += 1
            except:
                pass
        
        clear_all_caches()
        return True, f"{imported} imported, {updated} updated"
    except Exception as e:
        return False, f"CSV error: {str(e)}"

def format_time(seconds):
    """Format seconds as MM:SS"""
    if seconds is None:
        return "0:00"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"

# =============================================================================
# UI - TITLE AND SETTINGS BUTTON
# =============================================================================

# Title row with Settings button
col_title, col_settings = st.columns([6, 1])
with col_title:
    st.title("Speaker Assignments")
with col_settings:
    st.write("")  # Spacer
    if st.button("Settings", key="settings_btn"):
        st.session_state['show_settings_panel'] = not st.session_state.get('show_settings_panel', False)

# =============================================================================
# SETTINGS PANEL (collapsible, replaces sidebar)
# =============================================================================

if st.session_state.get('show_settings_panel', False):
    with st.container():
        st.markdown("---")
        settings_col1, settings_col2, settings_col3 = st.columns(3)
        
        with settings_col1:
            st.subheader("Embeddings Status")
            stored_embeddings, pending_embeddings = get_embedding_counts_cached()
            st.metric("Stored / Total", f"{stored_embeddings} / {stored_embeddings + pending_embeddings}")
            if pending_embeddings > 0:
                st.caption(f"{pending_embeddings} being computed...")
            else:
                st.caption("All embeddings computed")
        
        with settings_col2:
            st.subheader("Matching Settings")
            threshold = st.slider(
                "Threshold",
                min_value=0.5,
                max_value=0.95,
                value=st.session_state.get('similarity_threshold', SIMILARITY_THRESHOLD),
                step=0.05,
                help="Segments with higher similarity will be auto-assigned"
            )
            st.session_state['similarity_threshold'] = threshold
        
        with settings_col3:
            st.subheader("Import Speakers")
            with st.expander("CSV Import"):
                st.caption("Columns: name, email, department, company, meeting_count")
                csv_text = st.text_area(
                    "CSV Content",
                    height=100,
                    placeholder="name,email,meeting_count\nJohn Doe,john@example.com,5",
                    key="csv_import_text_panel",
                    label_visibility="collapsed"
                )
                if st.button("Import CSV", key="import_csv_panel", type="primary"):
                    if csv_text and csv_text.strip():
                        success, msg = import_speakers_from_csv(csv_text)
                        if success:
                            st.success(msg)
                            st.experimental_rerun()
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please paste CSV data")
        
        st.markdown("---")

# Get threshold from session state
SIMILARITY_THRESHOLD = st.session_state.get('similarity_threshold', SIMILARITY_THRESHOLD)

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Load speakers once (cached)
speakers = get_speakers_cached()

# Initialize filter state
if 'show_only_incomplete' not in st.session_state:
    st.session_state.show_only_incomplete = True

# Filter toggle
filter_col1, filter_col2 = st.columns([3, 1])
with filter_col1:
    st.subheader("Calls")
with filter_col2:
    show_filter = st.selectbox(
        "Filter",
        options=["Incomplete only", "All Calls"],
        index=0 if st.session_state.show_only_incomplete else 1,
        key="call_filter",
        label_visibility="collapsed"
    )
    st.session_state.show_only_incomplete = (show_filter == "Incomplete only")

calls = get_calls_cached(only_incomplete=st.session_state.show_only_incomplete)

if not calls:
    if st.session_state.show_only_incomplete:
        st.success("All calls are fully classified!")
        st.info("Select 'All Calls' to view completed calls.")
    else:
        st.info("No calls found. Upload a call via the CLI.")
else:
    # Initialize session state for call selection
    if 'selected_call_id' not in st.session_state:
        st.session_state.selected_call_id = calls[0][0]
    
    # Find current index
    current_idx = 0
    for i, m in enumerate(calls):
        if m[0] == st.session_state.selected_call_id:
            current_idx = i
            break
    
    if current_idx == 0 and calls[0][0] != st.session_state.selected_call_id:
        st.session_state.selected_call_id = calls[0][0]
    
    # Call selector with Edit and Delete buttons
    st.caption("Select Call")
    selector_col, edit_col, delete_col = st.columns([6, 1, 1])
    
    with selector_col:
        call_options = [f"{m[1]} ({m[3]})" for m in calls]
        selected_idx = st.selectbox(
            "Select Call", 
            range(len(calls)), 
            index=current_idx,
            format_func=lambda i: call_options[i],
            key="call_selector",
            label_visibility="collapsed"
        )
    
    with edit_col:
        if st.button("Edit", key="edit_call_btn", use_container_width=True):
            st.session_state['show_edit_call_dialog'] = True
    
    with delete_col:
        if st.button("Delete", key="delete_call_btn", type="secondary", use_container_width=True):
            st.session_state['show_delete_call_dialog'] = True
    
    st.session_state.selected_call_id = calls[selected_idx][0]
    selected_call = calls[selected_idx]
    call_id = selected_call[0]
    call_title = selected_call[1]
    
    # Edit Call Dialog
    if st.session_state.get('show_edit_call_dialog', False):
        with st.container():
            st.markdown("---")
            st.subheader("Edit Call Title")
            new_title = st.text_input("New Title", value=call_title, key="edit_call_title_input")
            edit_col1, edit_col2 = st.columns(2)
            with edit_col1:
                if st.button("Save", key="save_call_title_btn", type="primary", use_container_width=True):
                    if new_title and new_title.strip():
                        success, error = update_call_title(call_id, new_title.strip())
                        if success:
                            st.success("Title updated!")
                            st.session_state['show_edit_call_dialog'] = False
                            st.experimental_rerun()
                        else:
                            st.error(f"Error: {error}")
                    else:
                        st.warning("Title cannot be empty")
            with edit_col2:
                if st.button("Cancel", key="cancel_edit_call_btn", use_container_width=True):
                    st.session_state['show_edit_call_dialog'] = False
                    st.experimental_rerun()
            st.markdown("---")
    
    # Delete Call Dialog
    if st.session_state.get('show_delete_call_dialog', False):
        with st.container():
            st.markdown("---")
            st.subheader("Delete Call")
            st.warning(f"Are you sure you want to delete **{call_title}**?")
            st.caption("This will delete the call, all its contributions, and the MP3 file. Speaker voiceprints will be preserved.")
            del_col1, del_col2 = st.columns(2)
            with del_col1:
                if st.button("Yes, Delete", key="confirm_delete_call_btn", type="primary", use_container_width=True):
                    success, error = delete_call(call_id)
                    if success:
                        st.success("Call deleted!")
                        st.session_state['show_delete_call_dialog'] = False
                        st.session_state.pop('selected_call_id', None)
                        st.experimental_rerun()
                    else:
                        st.error(f"Error: {error}")
            with del_col2:
                if st.button("Cancel", key="cancel_delete_call_btn", use_container_width=True):
                    st.session_state['show_delete_call_dialog'] = False
                    st.experimental_rerun()
            st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Contributions", selected_call[4])
    with col2:
        st.metric("Detected Voices", selected_call[5])
    with col3:
        st.metric("Status", selected_call[3])
    
    st.divider()
    
    # Get diarization groups (cached)
    groups = get_diarization_groups_cached(call_id)
    
    if not groups:
        st.warning("No contributions found for this call.")
    else:
        # Audio player at top (separate component)
        audio_url = get_audio_url_cached(call_id)
        if audio_url:
            audio_html = f"""
            <div style="background:#1e1e1e;padding:15px;border-radius:10px;">
                <audio id="main_audio" controls style="width:100%;" preload="auto">
                    <source src="{audio_url}" type="audio/mpeg">
                </audio>
            </div>
            <script>
                // Use call-specific localStorage keys
                var callId = '{call_id}';
                var seekTimeKey = 'audioSeekTime_' + callId;
                var seekSecondsKey = 'audioSeekSeconds_' + callId;
                var positionKey = 'audioPosition_' + callId;
                
                // Track last processed seek to avoid duplicate seeks
                var lastSeekTime = localStorage.getItem(seekTimeKey) || '0';
                
                var audio = document.getElementById('main_audio');
                if (audio) {{
                    // Restore saved position after page reload
                    var savedPosition = localStorage.getItem(positionKey);
                    if (savedPosition && parseFloat(savedPosition) > 0) {{
                        audio.addEventListener('loadedmetadata', function() {{
                            audio.currentTime = parseFloat(savedPosition);
                        }}, {{once: true}});
                    }}
                    
                    // Save position periodically so it survives reruns
                    setInterval(function() {{
                        if (audio.currentTime > 0 && !audio.paused) {{
                            localStorage.setItem(positionKey, audio.currentTime.toString());
                        }}
                    }}, 500);
                    
                    // Also save on pause
                    audio.addEventListener('pause', function() {{
                        localStorage.setItem(positionKey, audio.currentTime.toString());
                    }});
                }}
                
                // Listen for seek commands via localStorage (from segment links)
                setInterval(function() {{
                    var currentSeekTime = localStorage.getItem(seekTimeKey) || '0';
                    if (currentSeekTime !== lastSeekTime) {{
                        lastSeekTime = currentSeekTime;
                        var seconds = parseFloat(localStorage.getItem(seekSecondsKey) || '0');
                        var audio = document.getElementById('main_audio');
                        if (audio) {{
                            // Wait for audio to be ready
                            if (audio.readyState >= 1) {{
                                audio.currentTime = seconds;
                                audio.play();
                                localStorage.setItem(positionKey, seconds.toString());
                            }} else {{
                                audio.addEventListener('loadedmetadata', function() {{
                                    audio.currentTime = seconds;
                                    audio.play();
                                    localStorage.setItem(positionKey, seconds.toString());
                                }}, {{once: true}});
                            }}
                        }}
                    }}
                }}, 100);
            </script>
            """
            components.html(audio_html, height=80)
        
        st.subheader("Assign Speakers")
        
        # Speaker options (use cached speakers)
        speaker_names = ["-- Not assigned --"] + [f"{s[1]} ({s[3]})" for s in speakers]
        speaker_display_to_id = {f"{s[1]} ({s[3]})": s[0] for s in speakers}
        speaker_id_to_display = {s[0]: f"{s[1]} ({s[3]})" for s in speakers}
        
        assignments_made = 0
        total_groups = len(groups)
        
        for group in groups:
            label = group[0]
            count = group[1]
            duration = group[2] or 0
            current_speaker = group[4]
            
            is_assigned = current_speaker is not None
            if is_assigned:
                assignments_made += 1
            
            # Get segments for this label - always fetch
            segments = get_segments_for_label(call_id, label)
            
            with st.container():
                # Header row with label and stats - show actual loaded count
                actual_count = len(segments) if segments else 0
                # Debug: show time range
                if segments:
                    first_time = format_time(segments[0][2] or 0)
                    last_time = format_time(segments[-1][2] or 0)
                    time_range = f"({first_time} - {last_time})"
                else:
                    time_range = ""
                if is_assigned:
                    st.markdown(f"**{label}** :green[assigned] - {actual_count} segments {time_range} | {duration:.0f}s total")
                else:
                    st.markdown(f"**{label}** :orange[unassigned] - {actual_count} segments {time_range} | {duration:.0f}s total")
                
                # Clickable segment links - first 20 words of ALL segments (scrollable)
                if segments and audio_url:
                    links_html = ""
                    for i, seg in enumerate(segments):  # Show ALL segments
                        start_sec = seg[2] or 0
                        time_str = format_time(start_sec)
                        text = seg[4] or ""
                        # Get first 20 words
                        words = text.split()[:20]
                        preview = " ".join(words)
                        if len(text.split()) > 20:
                            preview += "..."
                        # Escape for HTML
                        preview = preview.replace('&', '&amp;').replace('"', '&quot;').replace("'", "&#39;").replace("<", "&lt;").replace(">", "&gt;")
                        links_html += f'<a href="javascript:void(0);" onclick="seekTo({start_sec});" style="color:#4da6ff;text-decoration:none;display:block;padding:4px 0;font-size:13px;border-bottom:1px solid #333;"><b>[{time_str}]</b> {preview}</a>'
                    
                    # Fixed height with scroll for all segments
                    segment_html = f"""
                    <div style="max-height:150px;overflow-y:auto;padding:8px;background:#1a1a1a;border-radius:5px;margin:5px 0;">
                        {links_html}
                    </div>
                    <script>
                        var callId = '{call_id}';
                        function seekTo(seconds) {{
                            localStorage.setItem('audioSeekSeconds_' + callId, seconds.toString());
                            localStorage.setItem('audioSeekTime_' + callId, Date.now().toString());
                        }}
                    </script>
                    """
                    components.html(segment_html, height=170)
                
                # Action row: dropdown, assign, edit, new speaker, delete
                col_select, col_assign, col_edit, col_new, col_delete = st.columns([4, 1, 1, 1, 1])
                
                with col_select:
                    current_speaker_display = speaker_id_to_display.get(current_speaker, "-- Not assigned --") if current_speaker else "-- Not assigned --"
                    
                    selected_speaker_display = st.selectbox(
                        f"Speaker for {label}",
                        options=speaker_names,
                        index=speaker_names.index(current_speaker_display) if current_speaker_display in speaker_names else 0,
                        key=f"speaker_{label}",
                        label_visibility="collapsed"
                    )
                    
                    new_speaker_id = speaker_display_to_id.get(selected_speaker_display) if selected_speaker_display != "-- Not assigned --" else None
                
                with col_assign:
                    if new_speaker_id and new_speaker_id != current_speaker:
                        if st.button("Assign", key=f"assign_{label}", type="primary"):
                            result = assign_speaker_fast(call_id, label, new_speaker_id)
                            
                            if result['success']:
                                st.success("Assigned!")
                            else:
                                st.error(f"Error: {', '.join(result['errors'])}")
                    else:
                        st.button("Assign", key=f"assign_{label}", disabled=True)
                
                with col_edit:
                    # Edit button - only enabled if a speaker is selected
                    if new_speaker_id:
                        if st.button("Edit", key=f"edit_speaker_{label}", help="Edit speaker details"):
                            st.session_state[f'show_edit_speaker_dialog_{label}'] = new_speaker_id
                    else:
                        st.button("Edit", key=f"edit_speaker_{label}", disabled=True)
                
                with col_new:
                    # New Speaker button - opens dialog
                    if st.button("+ New", key=f"new_speaker_{label}", help="Create new speaker"):
                        st.session_state[f'show_new_speaker_dialog_{label}'] = True
                
                with col_delete:
                    if st.button("Delete", key=f"delete_{label}", help=f"Delete {label}"):
                        st.session_state[f'confirm_delete_{label}'] = True
                
                # Edit Speaker Dialog
                edit_speaker_id = st.session_state.get(f'show_edit_speaker_dialog_{label}')
                if edit_speaker_id:
                    speaker_details = get_speaker_details(edit_speaker_id)
                    if speaker_details:
                        with st.expander(f"Edit Speaker: {speaker_details['display_name']}", expanded=True):
                            with st.form(f"edit_speaker_form_{label}"):
                                edit_name = st.text_input("Name *", value=speaker_details['display_name'] or "", key=f"edit_name_{label}")
                                
                                col_email, col_dept = st.columns(2)
                                with col_email:
                                    edit_email = st.text_input("Email", value=speaker_details['email'] or "", key=f"edit_email_{label}")
                                with col_dept:
                                    edit_dept = st.text_input("Department", value=speaker_details['department'] or "", key=f"edit_dept_{label}")
                                
                                col_company, col_notes = st.columns(2)
                                with col_company:
                                    edit_company = st.text_input("Company", value=speaker_details['company'] or "", key=f"edit_company_{label}")
                                with col_notes:
                                    edit_notes = st.text_input("Notes", value=speaker_details['notes'] or "", key=f"edit_notes_{label}")
                                
                                col_submit, col_cancel = st.columns(2)
                                with col_submit:
                                    save_btn = st.form_submit_button("Save Changes", type="primary")
                                with col_cancel:
                                    cancel_edit_btn = st.form_submit_button("Cancel")
                                
                                if save_btn:
                                    if edit_name:
                                        success = update_speaker(
                                            edit_speaker_id,
                                            edit_name,
                                            edit_email if edit_email else None,
                                            edit_dept if edit_dept else None,
                                            edit_company if edit_company else None,
                                            edit_notes if edit_notes else None
                                        )
                                        if success:
                                            st.success(f"Speaker '{edit_name}' updated!")
                                            del st.session_state[f'show_edit_speaker_dialog_{label}']
                                            # No rerun - keeps audio player intact
                                    else:
                                        st.error("Name is required")
                                
                                if cancel_edit_btn:
                                    del st.session_state[f'show_edit_speaker_dialog_{label}']
                                    # No rerun - keeps audio player intact
                
                # New Speaker Dialog
                if st.session_state.get(f'show_new_speaker_dialog_{label}', False):
                    with st.expander("Create New Speaker", expanded=True):
                        with st.form(f"new_speaker_form_{label}"):
                            new_name = st.text_input("Name *", key=f"new_name_{label}")
                            
                            col_email, col_dept = st.columns(2)
                            with col_email:
                                new_email = st.text_input("Email", key=f"new_email_{label}")
                            with col_dept:
                                new_dept = st.text_input("Department", key=f"new_dept_{label}")
                            
                            col_company, col_notes = st.columns(2)
                            with col_company:
                                new_company = st.text_input("Company", key=f"new_company_{label}")
                            with col_notes:
                                new_notes = st.text_input("Notes", key=f"new_notes_{label}")
                            
                            col_submit, col_cancel = st.columns(2)
                            with col_submit:
                                create_btn = st.form_submit_button("Create & Assign", type="primary")
                            with col_cancel:
                                cancel_btn = st.form_submit_button("Cancel")
                            
                            if create_btn:
                                if new_name:
                                    speaker_id = create_speaker(
                                        new_name,
                                        new_email if new_email else None,
                                        new_dept if new_dept else None,
                                        new_company if new_company else None,
                                        new_notes if new_notes else None
                                    )
                                    if speaker_id:
                                        # Immediately assign to this label
                                        result = assign_speaker_fast(call_id, label, speaker_id)
                                        if result['success']:
                                            st.success(f"{new_name} created and assigned!")
                                            del st.session_state[f'show_new_speaker_dialog_{label}']
                                        else:
                                            st.error(f"Assignment error: {', '.join(result['errors'])}")
                                else:
                                    st.error("Name is required")
                            
                            if cancel_btn:
                                del st.session_state[f'show_new_speaker_dialog_{label}']
                
                # Delete confirmation
                if st.session_state.get(f'confirm_delete_{label}', False):
                    st.warning(f"Really delete {count} contributions from {label}?")
                    c1, c2, c3 = st.columns([1, 1, 4])
                    with c1:
                        if st.button("Yes, delete", key=f"yes_{label}", type="primary"):
                            success, message = delete_contributions(call_id, label)
                            if success:
                                st.success(message)
                                del st.session_state[f'confirm_delete_{label}']
                            else:
                                st.error(message)
                    with c2:
                        if st.button("Cancel", key=f"no_{label}"):
                            del st.session_state[f'confirm_delete_{label}']
                            # No rerun - keeps audio player intact
                
                st.divider()
        
        # Progress bar
        progress = assignments_made / total_groups if total_groups > 0 else 0
        st.progress(progress, text=f"Progress: {assignments_made}/{total_groups} speakers assigned")
        
        if assignments_made == total_groups:
            st.success("All speakers assigned!")
            if st.button("Refresh Page"):
                clear_all_caches()
                clear_all_caches()
                st.experimental_rerun()

# Footer
st.markdown("---")
st.caption("Speaker Classification with Voice Matching")
