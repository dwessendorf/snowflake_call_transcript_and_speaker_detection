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

# =============================================================================
# CACHED QUERY FUNCTIONS - Use caching to avoid repeated queries
# =============================================================================

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_speakers_cached():
    """Get all registered speakers, sorted by meeting count descending - CACHED"""
    results = session.sql("""
        SELECT speaker_id, display_name, email, COALESCE(meeting_count, 0) as meeting_count 
        FROM SPEAKERS 
        ORDER BY meeting_count DESC, display_name ASC
    """).collect()
    return [(r[0], r[1], r[2], r[3]) for r in results]

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_calls_cached(only_incomplete=True):
    """Get calls - CACHED"""
    filter_clause = "WHERE classification_status != 'completed'" if only_incomplete else ""
    results = session.sql(f"""
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
        ORDER BY m.call_date DESC
    """).collect()
    return [(r[0], r[1], r[2], r[3], r[4] or 0, r[5] or 0) for r in results]

@st.cache_data(ttl=30)
def get_diarization_groups_cached(call_id):
    """Get contributions grouped by diarization label - CACHED"""
    results = session.sql(f"""
        SELECT 
            diarization_label,
            COUNT(*) as segment_count,
            SUM(end_time_seconds - start_time_seconds) as total_duration,
            LISTAGG(SUBSTR(text_content, 1, 100), ' | ') WITHIN GROUP (ORDER BY segment_number) as sample_text,
            MIN(identified_speaker_id) as current_speaker_id
        FROM CALL_CONTRIBUTIONS
        WHERE call_id = '{call_id}'
        GROUP BY diarization_label
        ORDER BY MIN(segment_number)
    """).collect()
    return [(r[0], r[1], r[2], r[3], r[4]) for r in results]

@st.cache_data(ttl=60)
def get_embedding_counts_cached():
    """Get embedding counts - CACHED"""
    try:
        result = session.sql("""
            SELECT 
                (SELECT COUNT(*) FROM CONTRIBUTION_EMBEDDINGS) as stored,
                (SELECT COUNT(*) FROM CALL_CONTRIBUTIONS 
                 WHERE identified_speaker_id IS NULL 
                 AND duration_seconds >= 5.0
                 AND contribution_id NOT IN (SELECT contribution_id FROM CONTRIBUTION_EMBEDDINGS)) as pending
        """).collect()
        return result[0][0] or 0, result[0][1] or 0
    except:
        return 0, 0

@st.cache_data(ttl=300)  # Cache audio URL for 5 minutes
def get_audio_url_cached(call_id):
    """Get presigned URL for call audio - CACHED"""
    try:
        result = session.sql(f"SELECT recording_path FROM CALLS WHERE call_id = '{call_id}'").collect()
        if not result or not result[0][0]:
            return None
        recording_path = result[0][0]
        if recording_path.startswith('@'):
            recording_path = recording_path[1:]
        parts = recording_path.split('/')
        file_path = '/'.join(parts[1:]) if len(parts) > 1 else parts[0]
        
        url_result = session.sql(f"SELECT GET_PRESIGNED_URL(@{parts[0]}, '{file_path}', 3600) as url").collect()
        return url_result[0][0] if url_result else None
    except:
        return None

# =============================================================================
# NON-CACHED FUNCTIONS - For data that changes frequently or needs fresh data
# =============================================================================

def run_query(sql):
    """Execute SQL and return results as list of tuples"""
    return session.sql(sql).collect()

def get_segments_for_label(call_id, diarization_label):
    """Get individual segments for a diarization label"""
    results = session.sql(f"""
        SELECT 
            contribution_id,
            segment_number,
            start_time_seconds,
            end_time_seconds,
            text_content
        FROM CALL_CONTRIBUTIONS
        WHERE call_id = '{call_id}'
        AND diarization_label = '{diarization_label}'
        ORDER BY segment_number
    """).collect()
    return [(r[0], r[1], r[2], r[3], r[4]) for r in results]

def extract_embedding_via_procedure(call_id, contribution_id):
    """Extract embedding using the Snowflake procedure"""
    try:
        results = run_query(f"CALL EXTRACT_CONTRIBUTION_EMBEDDING('{call_id}', '{contribution_id}')")
        result = results[0] if results else None
        
        if result and result[0]:
            data = result[0]
            if isinstance(data, str):
                data = json.loads(data)
            
            if data.get('status') == 'success':
                return data.get('embedding'), None
            else:
                return None, data.get('message', 'Unknown error')
        
        return None, "No result from procedure"
    except Exception as e:
        return None, str(e)

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

def match_voiceprint_against_embeddings(speaker_id, source_call_id, threshold=0.75):
    """Match a speaker's voiceprint against pre-computed embeddings in other calls."""
    try:
        results = run_query(f"""
            CALL MATCH_VOICEPRINT_AGAINST_EMBEDDINGS('{speaker_id}', '{source_call_id}', {threshold})
        """)
        if results and results[0][0]:
            data = results[0][0]
            if isinstance(data, str):
                data = json.loads(data)
            return data
        return {'message': 'No result from matching procedure'}
    except Exception as e:
        return {'errors': [str(e)]}

def assign_speaker_with_matching(call_id, diarization_label, speaker_id, speaker_name, threshold):
    """Assign a speaker to a diarization label."""
    MIN_DURATION_FOR_EMBEDDING = 5.0
    
    results = {
        'success': False,
        'embedding_extracted': False,
        'voiceprint_saved': False,
        'segments_tested': 0,
        'auto_classified': 0,
        'kept_for_review': 0,
        'skipped_short': 0,
        'errors': []
    }
    
    try:
        # Check if speaker already in call
        existing = run_query(f"""
            SELECT COUNT(*) FROM CALL_CONTRIBUTIONS
            WHERE call_id = '{call_id}' 
            AND identified_speaker_id = '{speaker_id}'
            AND classification_status = 'classified'
        """)
        speaker_already_in_call = existing[0][0] > 0 if existing else False
        
        # Get contributions for this label
        contributions = run_query(f"""
            SELECT contribution_id, duration_seconds
            FROM CALL_CONTRIBUTIONS
            WHERE call_id = '{call_id}' 
            AND diarization_label = '{diarization_label}'
            ORDER BY segment_number
        """)
        
        if not contributions:
            results['errors'].append('No contributions found')
            return results
        
        total_count = len(contributions)
        long_segments = [(c[0], c[1]) for c in contributions if (c[1] or 0) >= MIN_DURATION_FOR_EMBEDDING]
        short_segments = [(c[0], c[1]) for c in contributions if (c[1] or 0) < MIN_DURATION_FOR_EMBEDDING]
        
        results['skipped_short'] = len(short_segments)
        
        # Extract embedding from longest segment
        if long_segments:
            long_segments.sort(key=lambda x: x[1] or 0, reverse=True)
            best_contribution_id = long_segments[0][0]
            embedding, error = extract_embedding_via_procedure(call_id, best_contribution_id)
            if embedding:
                results['embedding_extracted'] = True
            elif error:
                results['errors'].append(f"Embedding extraction: {error}")
        
        # Assign speaker
        session.sql(f"""
            UPDATE CALL_CONTRIBUTIONS
            SET identified_speaker_id = '{speaker_id}',
                identification_method = 'manual',
                identification_confidence = 1.0,
                classification_status = 'classified'
            WHERE call_id = '{call_id}' 
            AND diarization_label = '{diarization_label}'
        """).collect()
        
        # Remove from queue
        session.sql(f"""
            DELETE FROM CLASSIFICATION_QUEUE 
            WHERE call_id = '{call_id}'
            AND diarization_label = '{diarization_label}'
        """).collect()
        
        # Increment meeting count
        if not speaker_already_in_call:
            increment_speaker_meeting_count(speaker_id)
        
        results['success'] = True
        results['auto_classified'] = total_count
        
        # Background matching
        try:
            match_result = match_voiceprint_against_embeddings(speaker_id, call_id, threshold)
            if match_result.get('contributions_updated', 0) > 0:
                results['background_matches'] = match_result.get('contributions_updated', 0)
                results['background_calls'] = len(match_result.get('calls_affected', []))
            if match_result.get('voiceprint_created'):
                results['voiceprint_saved'] = True
        except Exception as e:
            results['errors'].append(f"Background matching: {str(e)}")
        
        # Clear caches after assignment
        get_diarization_groups_cached.clear()
        get_speakers_cached.clear()
        
        return results
        
    except Exception as e:
        results['errors'].append(str(e))
        return results

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
            return False, "Keine Beiträge gefunden"
        
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
        
        get_diarization_groups_cached.clear()
        return True, f"{count} Beiträge gelöscht"
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
        get_speakers_cached.clear()
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
        get_speakers_cached.clear()
        return True
    except Exception as e:
        st.error(f"Error updating speaker: {e}")
        return False

def delete_speaker(speaker_id):
    """Delete a speaker (only if not used)"""
    try:
        count_result = run_query(f"""
            SELECT COUNT(*) FROM CALL_CONTRIBUTIONS
            WHERE identified_speaker_id = '{speaker_id}'
        """)
        count = count_result[0][0] if count_result else 0
        
        if count > 0:
            return False, f"Sprecher wird in {count} Beiträgen verwendet"
        
        session.sql(f"DELETE FROM SPEAKERS WHERE speaker_id = '{speaker_id}'").collect()
        get_speakers_cached.clear()
        return True, None
    except Exception as e:
        return False, str(e)

def update_call_status(call_id):
    """Update call status based on classification progress"""
    results = run_query(f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN classification_status IN ('classified', 'auto_classified') THEN 1 ELSE 0 END) as classified
        FROM CALL_CONTRIBUTIONS
        WHERE call_id = '{call_id}'
    """)
    
    if results:
        total, classified = results[0][0], results[0][1]
        if total == classified and total > 0:
            session.sql(f"""
                UPDATE CALLS SET classification_status = 'completed'
                WHERE call_id = '{call_id}'
            """).collect()
            get_calls_cached.clear()

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
            return False, "CSV muss eine 'name' oder 'display_name' Spalte haben"
        
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
        
        get_speakers_cached.clear()
        return True, f"✅ {imported} neu importiert, {updated} aktualisiert"
    except Exception as e:
        return False, f"CSV-Fehler: {str(e)}"

# =============================================================================
# UI - TITLE
# =============================================================================

st.title("🎤 Speaker Classification")

# =============================================================================
# SIDEBAR - Speaker Management
# =============================================================================

st.sidebar.header("👥 Sprecher")

# Load speakers once (cached)
speakers = get_speakers_cached()

if speakers:
    st.sidebar.write(f"**{len(speakers)} Sprecher registriert**")
    
    speaker_search = st.sidebar.text_input(
        "🔍 Sprecher suchen",
        placeholder="Name oder E-Mail...",
        key="sidebar_speaker_search"
    )
    
    filtered_speakers = filter_speakers(speakers, speaker_search)
    
    if speaker_search and not filtered_speakers:
        st.sidebar.info(f"Keine Sprecher gefunden für '{speaker_search}'")
    elif filtered_speakers:
        if speaker_search:
            st.sidebar.caption(f"{len(filtered_speakers)} Treffer")
        
        # Only show first 10 for performance
        display_speakers = filtered_speakers[:10]
        if len(filtered_speakers) > 10:
            st.sidebar.caption(f"Zeige erste 10 von {len(filtered_speakers)}")
        
        for s in display_speakers:
            speaker_id, speaker_name, speaker_email, meeting_count = s[0], s[1], s[2], s[3]
            
            with st.sidebar.expander(f"📝 {speaker_name} ({meeting_count})"):
                details = get_speaker_details(speaker_id)
                if details:
                    _, name, email, dept, company, notes, is_internal, mtg_count = details
                    
                    with st.form(f"edit_speaker_{speaker_id}"):
                        edit_name = st.text_input("Name *", value=name or "")
                        edit_email = st.text_input("E-Mail", value=email or "")
                        edit_dept = st.text_input("Abteilung", value=dept or "")
                        edit_company = st.text_input("Firma", value=company or "")
                        edit_meeting_count = st.number_input("Meetings", value=mtg_count or 0, min_value=0)
                        edit_notes = st.text_area("Notizen", value=notes or "")
                        edit_internal = st.checkbox("Intern", value=is_internal if is_internal is not None else True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            save_btn = st.form_submit_button("💾 Speichern")
                        with col2:
                            delete_btn = st.form_submit_button("🗑️ Löschen")
                        
                        if save_btn:
                            if edit_name:
                                success = update_speaker(
                                    speaker_id, edit_name,
                                    edit_email if edit_email else None,
                                    edit_dept if edit_dept else None,
                                    edit_company if edit_company else None,
                                    edit_notes if edit_notes else None,
                                    edit_internal, edit_meeting_count
                                )
                                if success:
                                    st.success("✅ Gespeichert!")
                                    st.experimental_rerun()
                            else:
                                st.error("Name ist erforderlich")
                        
                        if delete_btn:
                            success, error = delete_speaker(speaker_id)
                            if success:
                                st.success("✅ Gelöscht!")
                                st.experimental_rerun()
                            else:
                                st.error(f"❌ {error}")

st.sidebar.divider()

# CSV Import
st.sidebar.subheader("📤 CSV Import")
with st.sidebar.expander("CSV einfügen"):
    st.caption("Spalten: name, email, department, company, meeting_count")
    csv_text = st.text_area(
        "CSV Inhalt",
        height=150,
        placeholder="name,email,meeting_count\nMax Mustermann,max@example.com,5\n...",
        key="csv_import_text"
    )
    if st.button("📥 Importieren", key="import_csv_btn"):
        if csv_text and csv_text.strip():
            success, msg = import_speakers_from_csv(csv_text)
            if success:
                st.success(msg)
                st.experimental_rerun()
            else:
                st.error(msg)
        else:
            st.warning("Bitte CSV-Daten einfügen")

st.sidebar.divider()
st.sidebar.subheader("➕ Neuen Sprecher anlegen")

with st.sidebar.form("new_speaker"):
    new_name = st.text_input("Name *")
    new_email = st.text_input("E-Mail")
    new_dept = st.text_input("Abteilung")
    new_company = st.text_input("Firma")
    new_notes = st.text_area("Notizen")
    
    submitted = st.form_submit_button("Sprecher anlegen")
    if submitted:
        if new_name:
            speaker_id = create_speaker(
                new_name, 
                new_email if new_email else None,
                new_dept if new_dept else None,
                new_company if new_company else None,
                new_notes if new_notes else None
            )
            if speaker_id:
                st.sidebar.success(f"✅ {new_name} angelegt!")
                st.experimental_rerun()
        else:
            st.sidebar.error("Name ist erforderlich")

# Settings
st.sidebar.divider()
st.sidebar.subheader("⚙️ Einstellungen")
threshold = st.sidebar.slider(
    "Matching-Schwellenwert",
    min_value=0.5,
    max_value=0.95,
    value=SIMILARITY_THRESHOLD,
    step=0.05,
    help="Segmente mit höherer Ähnlichkeit werden automatisch zugeordnet"
)
SIMILARITY_THRESHOLD = threshold

# Background Matching Info
st.sidebar.divider()
st.sidebar.subheader("🔄 Auto-Matching")
stored_embeddings, pending_embeddings = get_embedding_counts_cached()
st.sidebar.metric("Gespeicherte Embeddings", stored_embeddings)
st.sidebar.metric("Wartende Beiträge", pending_embeddings)
if pending_embeddings > 0:
    st.sidebar.caption("⏳ Embeddings werden automatisch berechnet (alle 5 Min)")
else:
    st.sidebar.caption("✅ Alle Embeddings berechnet")

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Initialize filter state
if 'show_only_incomplete' not in st.session_state:
    st.session_state.show_only_incomplete = True

# Filter toggle
filter_col1, filter_col2 = st.columns([3, 1])
with filter_col1:
    st.subheader("📋 Calls")
with filter_col2:
    show_filter = st.selectbox(
        "Filter",
        options=["Nur unvollständige", "Alle Calls"],
        index=0 if st.session_state.show_only_incomplete else 1,
        key="call_filter",
        label_visibility="collapsed"
    )
    st.session_state.show_only_incomplete = (show_filter == "Nur unvollständige")

calls = get_calls_cached(only_incomplete=st.session_state.show_only_incomplete)

if not calls:
    if st.session_state.show_only_incomplete:
        st.success("🎉 Alle Calls sind vollständig klassifiziert!")
        st.info("Wählen Sie 'Alle Calls' um abgeschlossene Calls anzuzeigen.")
    else:
        st.info("Keine Calls gefunden. Laden Sie ein Call über die CLI hoch.")
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
    
    # Call selector
    call_options = [f"{m[1]} ({m[3]})" for m in calls]
    selected_idx = st.selectbox(
        "Call auswählen", 
        range(len(calls)), 
        index=current_idx,
        format_func=lambda i: call_options[i],
        key="call_selector"
    )
    
    st.session_state.selected_call_id = calls[selected_idx][0]
    selected_call = calls[selected_idx]
    call_id = selected_call[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Beiträge", selected_call[4])
    with col2:
        st.metric("Erkannte Stimmen", selected_call[5])
    with col3:
        st.metric("Status", selected_call[3])
    
    st.divider()
    
    # Get diarization groups (cached)
    groups = get_diarization_groups_cached(call_id)
    
    if not groups:
        st.warning("Keine Beiträge für dieses Call gefunden.")
    else:
        # Audio player
        audio_url = get_audio_url_cached(call_id)
        if audio_url:
            buttons_html = ""
            for group in groups:
                label = group[0]
                segments = get_segments_for_label(call_id, label)
                if segments:
                    first_start_sec = segments[0][2] or 0
                    start_fmt = f"{int(first_start_sec//60)}:{int(first_start_sec%60):02d}"
                    buttons_html += f"""
                    <button onclick="seekTo({first_start_sec});" 
                            style="background: #4CAF50; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 11px; margin: 2px;">
                        {label} ({start_fmt})
                    </button>
                    """
            
            audio_html = f"""
            <div style="background: #1e1e1e; padding: 15px; border-radius: 10px;">
                <audio id="call_audio" controls style="width: 100%;" preload="metadata">
                    <source src="{audio_url}" type="audio/mpeg">
                </audio>
                <div style="margin-top: 10px;">
                    <span style="color: #888; font-size: 12px;">Springe zu: </span>
                    {buttons_html}
                </div>
            </div>
            <script>
                var audioElement = document.getElementById('call_audio');
                function seekTo(seconds) {{
                    audioElement.currentTime = seconds;
                    audioElement.play();
                }}
                window.parent.seekAudioTo = function(seconds) {{
                    audioElement.currentTime = seconds;
                    audioElement.play();
                }};
            </script>
            """
            components.html(audio_html, height=130)
        
        st.subheader("🎯 Sprecher zuordnen")
        st.caption(f"Schwellenwert für Auto-Matching: {SIMILARITY_THRESHOLD:.0%}")
        
        # Speaker options (use cached speakers)
        speaker_names = ["-- Nicht zugeordnet --"] + [f"{s[1]} ({s[3]})" for s in speakers]
        speaker_display_to_id = {f"{s[1]} ({s[3]})": s[0] for s in speakers}
        speaker_id_to_display = {s[0]: f"{s[1]} ({s[3]})" for s in speakers}
        
        assignments_made = 0
        total_groups = len(groups)
        
        for group in groups:
            label = group[0]
            count = group[1]
            duration = group[2] or 0
            current_speaker = group[4]
            
            segments = get_segments_for_label(call_id, label) if audio_url else []
            
            is_assigned = current_speaker is not None
            if is_assigned:
                assignments_made += 1
            
            with st.container():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if is_assigned:
                        st.success(f"**{label}** ✓")
                    else:
                        st.warning(f"**{label}**")
                    st.caption(f"{count} Segmente · {duration:.0f}s")
                
                with col2:
                    # Sample text
                    sample = (group[3] or "")[:300]
                    st.caption(f"*\"{sample}...\"*" if len(sample) >= 300 else f"*\"{sample}\"*")
                    
                    col_select, col_button, col_delete = st.columns([3, 1, 1])
                    
                    with col_select:
                        current_speaker_display = speaker_id_to_display.get(current_speaker, "-- Nicht zugeordnet --") if current_speaker else "-- Nicht zugeordnet --"
                        
                        selected_speaker_display = st.selectbox(
                            f"Sprecher für {label}",
                            options=speaker_names,
                            index=speaker_names.index(current_speaker_display) if current_speaker_display in speaker_names else 0,
                            key=f"speaker_{label}",
                            label_visibility="collapsed"
                        )
                        
                        new_speaker_id = speaker_display_to_id.get(selected_speaker_display) if selected_speaker_display != "-- Nicht zugeordnet --" else None
                    
                    with col_button:
                        if new_speaker_id and new_speaker_id != current_speaker:
                            if st.button(f"✓ Zuordnen", key=f"assign_{label}"):
                                with st.spinner(f"Verarbeite {label}..."):
                                    result = assign_speaker_with_matching(
                                        call_id, label, new_speaker_id, selected_speaker_display, SIMILARITY_THRESHOLD
                                    )
                                    
                                    if result['success']:
                                        msg = f"✅ Zugeordnet! {result['auto_classified']} Segmente"
                                        if result.get('background_matches', 0) > 0:
                                            msg += f"\n🔄 {result['background_matches']} in anderen Calls!"
                                        st.success(msg)
                                        update_call_status(call_id)
                                        st.experimental_rerun()
                                    else:
                                        st.error(f"Fehler: {', '.join(result['errors'])}")
                        elif not new_speaker_id:
                            st.button("✓ Zuordnen", key=f"assign_{label}", disabled=True)
                    
                    with col_delete:
                        if st.button("🗑️", key=f"delete_{label}", help=f"Lösche {label}"):
                            st.session_state[f'confirm_delete_{label}'] = True
                
                if st.session_state.get(f'confirm_delete_{label}', False):
                    st.warning(f"Wirklich {count} Beiträge löschen?")
                    c1, c2, c3 = st.columns([1, 1, 4])
                    with c1:
                        if st.button("✓ Ja", key=f"yes_{label}"):
                            success, message = delete_contributions(call_id, label)
                            if success:
                                st.success(message)
                                del st.session_state[f'confirm_delete_{label}']
                                update_call_status(call_id)
                                st.experimental_rerun()
                            else:
                                st.error(message)
                    with c2:
                        if st.button("✗ Nein", key=f"no_{label}"):
                            del st.session_state[f'confirm_delete_{label}']
                            st.experimental_rerun()
                
                st.divider()
        
        # Progress bar
        progress = assignments_made / total_groups if total_groups > 0 else 0
        st.progress(progress, text=f"Fortschritt: {assignments_made}/{total_groups} Sprecher zugeordnet")
        
        if assignments_made == total_groups:
            st.success("✅ Alle Sprecher zugeordnet!")
            if st.button("🔄 Seite aktualisieren"):
                get_calls_cached.clear()
                get_diarization_groups_cached.clear()
                st.experimental_rerun()

# Footer
st.markdown("---")
st.caption("Call Transcription - Speaker Classification mit Voice Matching")
