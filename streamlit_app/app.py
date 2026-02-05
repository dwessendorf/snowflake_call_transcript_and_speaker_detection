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
from snowflake.snowpark.context import get_active_session

st.set_page_config(page_title="Speaker Classification", page_icon="🎤", layout="wide")

st.title("🎤 Speaker Classification")

# Configuration
SIMILARITY_THRESHOLD = 0.75  # Threshold for auto-matching

# Get Snowpark session (works in Streamlit-in-Snowflake)
session = get_active_session()

def run_query(sql):
    """Execute SQL and return results as list of tuples"""
    return session.sql(sql).collect()

def run_query_df(sql):
    """Execute SQL and return results as pandas DataFrame"""
    return session.sql(sql).to_pandas()

def get_audio_url(meeting_id):
    """Get presigned URL for meeting audio"""
    try:
        result = run_query(f"SELECT recording_path FROM MEETINGS WHERE meeting_id = '{meeting_id}'")
        if not result or not result[0][0]:
            return None
        recording_path = result[0][0]
        # Parse stage and path
        if recording_path.startswith('@'):
            recording_path = recording_path[1:]
        parts = recording_path.split('/')
        stage_name = parts[0] if '.' in parts[0] else f"@{parts[0]}"
        file_path = '/'.join(parts[1:]) if len(parts) > 1 else parts[0]
        
        url_result = run_query(f"SELECT GET_PRESIGNED_URL(@{parts[0]}, '{file_path}', 3600) as url")
        return url_result[0][0] if url_result else None
    except Exception as e:
        return None

def get_meetings(only_incomplete=True):
    """Get meetings, optionally filtered to only incomplete ones"""
    filter_clause = "WHERE classification_status != 'completed'" if only_incomplete else ""
    results = run_query(f"""
        SELECT meeting_id, title, meeting_date, classification_status,
               (SELECT COUNT(*) FROM MEETING_CONTRIBUTIONS WHERE meeting_id = m.meeting_id) as total_contributions,
               (SELECT COUNT(DISTINCT diarization_label) FROM MEETING_CONTRIBUTIONS WHERE meeting_id = m.meeting_id) as speaker_count
        FROM MEETINGS m
        {filter_clause}
        ORDER BY meeting_date DESC
    """)
    return [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in results]

def get_speakers():
    """Get all registered speakers"""
    results = run_query("SELECT speaker_id, display_name, email FROM SPEAKERS ORDER BY display_name")
    return [(r[0], r[1], r[2]) for r in results]

def get_diarization_groups(meeting_id):
    """Get contributions grouped by diarization label"""
    results = run_query(f"""
        SELECT 
            diarization_label,
            COUNT(*) as segment_count,
            SUM(end_time_seconds - start_time_seconds) as total_duration,
            LISTAGG(SUBSTR(text_content, 1, 100), ' | ') WITHIN GROUP (ORDER BY segment_number) as sample_text,
            MIN(identified_speaker_id) as current_speaker_id
        FROM MEETING_CONTRIBUTIONS
        WHERE meeting_id = '{meeting_id}'
        GROUP BY diarization_label
        ORDER BY MIN(segment_number)
    """)
    return [(r[0], r[1], r[2], r[3], r[4]) for r in results]

def get_segments_for_label(meeting_id, diarization_label):
    """Get individual segments for a diarization label"""
    results = run_query(f"""
        SELECT 
            contribution_id,
            segment_number,
            start_time_seconds,
            end_time_seconds,
            text_content
        FROM MEETING_CONTRIBUTIONS
        WHERE meeting_id = '{meeting_id}'
        AND diarization_label = '{diarization_label}'
        ORDER BY segment_number
    """)
    return [(r[0], r[1], r[2], r[3], r[4]) for r in results]

def compute_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings"""
    if not embedding1 or not embedding2:
        return 0.0
    
    arr1 = np.array(embedding1)
    arr2 = np.array(embedding2)
    
    # Cosine similarity
    dot_product = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def extract_embedding_via_procedure(meeting_id, contribution_id):
    """Extract embedding using the Snowflake procedure"""
    try:
        results = run_query(f"CALL EXTRACT_CONTRIBUTION_EMBEDDING('{meeting_id}', '{contribution_id}')")
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

def assign_speaker_with_matching(meeting_id, diarization_label, speaker_id, speaker_name, threshold):
    """
    Assign a speaker to a diarization label.
    Only extracts embeddings for segments >= 5 seconds.
    Shorter segments are assigned without embedding extraction.
    """
    MIN_DURATION_FOR_EMBEDDING = 5.0  # seconds
    
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
        # Get all contributions for this label with their durations
        contributions = run_query(f"""
            SELECT contribution_id, duration_seconds
            FROM MEETING_CONTRIBUTIONS
            WHERE meeting_id = '{meeting_id}' 
            AND diarization_label = '{diarization_label}'
            ORDER BY segment_number
        """)
        
        if not contributions:
            results['errors'].append('No contributions found')
            return results
        
        total_count = len(contributions)
        
        # Find the longest segment >= 5 seconds for embedding extraction
        long_segments = [(c[0], c[1]) for c in contributions if (c[1] or 0) >= MIN_DURATION_FOR_EMBEDDING]
        short_segments = [(c[0], c[1]) for c in contributions if (c[1] or 0) < MIN_DURATION_FOR_EMBEDDING]
        
        results['skipped_short'] = len(short_segments)
        
        # Try to extract embedding from the longest segment (if any are long enough)
        if long_segments:
            # Sort by duration descending to get the longest
            long_segments.sort(key=lambda x: x[1] or 0, reverse=True)
            best_contribution_id = long_segments[0][0]
            best_duration = long_segments[0][1]
            
            # Extract embedding for the longest segment
            embedding, error = extract_embedding_via_procedure(meeting_id, best_contribution_id)
            if embedding:
                results['embedding_extracted'] = True
                # Could save voiceprint here if needed in the future
                results['voiceprint_saved'] = False
            elif error:
                results['errors'].append(f"Embedding extraction: {error}")
        
        # Assign speaker to ALL contributions with this diarization label
        session.sql(f"""
            UPDATE MEETING_CONTRIBUTIONS
            SET identified_speaker_id = '{speaker_id}',
                identification_method = 'manual',
                identification_confidence = 1.0,
                classification_status = 'classified'
            WHERE meeting_id = '{meeting_id}' 
            AND diarization_label = '{diarization_label}'
        """).collect()
        
        # Remove from classification queue
        session.sql(f"""
            DELETE FROM CLASSIFICATION_QUEUE 
            WHERE meeting_id = '{meeting_id}'
            AND diarization_label = '{diarization_label}'
        """).collect()
        
        results['success'] = True
        results['auto_classified'] = total_count
        
        return results
        
    except Exception as e:
        results['errors'].append(str(e))
        return results

def delete_contributions(meeting_id, diarization_label):
    """Delete all contributions for a specific diarization label from a meeting"""
    try:
        # Get count first
        count_result = run_query(f"""
            SELECT COUNT(*) FROM MEETING_CONTRIBUTIONS
            WHERE meeting_id = '{meeting_id}' 
            AND diarization_label = '{diarization_label}'
        """)
        count = count_result[0][0] if count_result else 0
        
        if count == 0:
            return False, "Keine Beiträge gefunden"
        
        # Delete from classification queue first
        session.sql(f"""
            DELETE FROM CLASSIFICATION_QUEUE 
            WHERE meeting_id = '{meeting_id}'
            AND diarization_label = '{diarization_label}'
        """).collect()
        
        # Delete contributions
        session.sql(f"""
            DELETE FROM MEETING_CONTRIBUTIONS
            WHERE meeting_id = '{meeting_id}' 
            AND diarization_label = '{diarization_label}'
        """).collect()
        
        return True, f"{count} Beiträge gelöscht"
    except Exception as e:
        return False, str(e)

def create_speaker(name, email=None, department=None, company=None, notes=None):
    """Create a new speaker"""
    speaker_id = f"SPK_{uuid.uuid4().hex[:16]}"
    
    try:
        # Build insert with proper escaping
        email_val = f"'{email}'" if email else 'NULL'
        dept_val = f"'{department}'" if department else 'NULL'
        company_val = f"'{company}'" if company else 'NULL'
        notes_val = f"'{notes}'" if notes else 'NULL'
        
        session.sql(f"""
            INSERT INTO SPEAKERS (speaker_id, display_name, email, department, company, notes, is_internal, created_at, updated_at, created_by)
            VALUES ('{speaker_id}', '{name}', {email_val}, {dept_val}, {company_val}, {notes_val}, TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_USER())
        """).collect()
        return speaker_id
    except Exception as e:
        st.error(f"Error creating speaker: {e}")
        return None

def get_speaker_details(speaker_id):
    """Get full details of a speaker"""
    results = run_query(f"""
        SELECT speaker_id, display_name, email, department, company, notes, is_internal
        FROM SPEAKERS
        WHERE speaker_id = '{speaker_id}'
    """)
    if results:
        r = results[0]
        return (r[0], r[1], r[2], r[3], r[4], r[5], r[6])
    return None

def update_speaker(speaker_id, name, email=None, department=None, company=None, notes=None, is_internal=True):
    """Update an existing speaker"""
    try:
        # Build update with proper escaping
        email_val = f"'{email}'" if email else 'NULL'
        dept_val = f"'{department}'" if department else 'NULL'
        company_val = f"'{company}'" if company else 'NULL'
        notes_escaped = notes.replace("'", "''") if notes else None
        notes_val = f"'{notes_escaped}'" if notes_escaped else 'NULL'
        internal_val = 'TRUE' if is_internal else 'FALSE'
        
        session.sql(f"""
            UPDATE SPEAKERS
            SET display_name = '{name}',
                email = {email_val},
                department = {dept_val},
                company = {company_val},
                notes = {notes_val},
                is_internal = {internal_val},
                updated_at = CURRENT_TIMESTAMP
            WHERE speaker_id = '{speaker_id}'
        """).collect()
        return True
    except Exception as e:
        st.error(f"Error updating speaker: {e}")
        return False

def delete_speaker(speaker_id):
    """Delete a speaker (only if not used in any meetings)"""
    try:
        # Check if speaker is used in any contributions
        count_result = run_query(f"""
            SELECT COUNT(*) FROM MEETING_CONTRIBUTIONS
            WHERE identified_speaker_id = '{speaker_id}'
        """)
        count = count_result[0][0] if count_result else 0
        
        if count > 0:
            return False, f"Sprecher wird in {count} Beiträgen verwendet"
        
        # Delete speaker
        session.sql(f"DELETE FROM SPEAKERS WHERE speaker_id = '{speaker_id}'").collect()
        return True, None
    except Exception as e:
        return False, str(e)

def update_meeting_status(meeting_id):
    """Update meeting status based on classification progress"""
    # Check if all contributions are classified (either manually or auto)
    results = run_query(f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN classification_status IN ('classified', 'auto_classified') THEN 1 ELSE 0 END) as classified
        FROM MEETING_CONTRIBUTIONS
        WHERE meeting_id = '{meeting_id}'
    """)
    
    if results:
        total, classified = results[0][0], results[0][1]
        if total == classified and total > 0:
            session.sql(f"""
                UPDATE MEETINGS SET classification_status = 'completed'
                WHERE meeting_id = '{meeting_id}'
            """).collect()

# Sidebar - Speaker Management
st.sidebar.header("👥 Sprecher")
speakers = get_speakers()

if speakers:
    st.sidebar.write(f"**{len(speakers)} Sprecher registriert:**")
    
    # Speaker list with edit option
    for s in speakers:
        speaker_id, speaker_name, speaker_email = s[0], s[1], s[2]
        
        with st.sidebar.expander(f"📝 {speaker_name}"):
            # Get full speaker details
            details = get_speaker_details(speaker_id)
            if details:
                _, name, email, dept, company, notes, is_internal = details
                
                with st.form(f"edit_speaker_{speaker_id}"):
                    edit_name = st.text_input("Name *", value=name or "")
                    edit_email = st.text_input("E-Mail", value=email or "")
                    edit_dept = st.text_input("Abteilung", value=dept or "")
                    edit_company = st.text_input("Firma", value=company or "")
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
                                speaker_id,
                                edit_name,
                                edit_email if edit_email else None,
                                edit_dept if edit_dept else None,
                                edit_company if edit_company else None,
                                edit_notes if edit_notes else None,
                                edit_internal
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

# Main content
# Initialize filter state
if 'show_only_incomplete' not in st.session_state:
    st.session_state.show_only_incomplete = True

# Filter toggle
filter_col1, filter_col2 = st.columns([3, 1])
with filter_col1:
    st.subheader("📋 Meetings")
with filter_col2:
    show_filter = st.selectbox(
        "Filter",
        options=["Nur unvollständige", "Alle Meetings"],
        index=0 if st.session_state.show_only_incomplete else 1,
        key="meeting_filter",
        label_visibility="collapsed"
    )
    st.session_state.show_only_incomplete = (show_filter == "Nur unvollständige")

meetings = get_meetings(only_incomplete=st.session_state.show_only_incomplete)

if not meetings:
    if st.session_state.show_only_incomplete:
        st.success("🎉 Alle Meetings sind vollständig klassifiziert!")
        st.info("Wählen Sie 'Alle Meetings' um abgeschlossene Meetings anzuzeigen.")
    else:
        st.info("Keine Meetings gefunden. Laden Sie ein Meeting über die CLI hoch.")
else:
    # Initialize session state for meeting selection
    if 'selected_meeting_id' not in st.session_state:
        st.session_state.selected_meeting_id = meetings[0][0]
    
    # Find current index based on stored meeting_id
    current_idx = 0
    for i, m in enumerate(meetings):
        if m[0] == st.session_state.selected_meeting_id:
            current_idx = i
            break
    
    # If stored meeting not in current list, reset to first
    if current_idx == 0 and meetings[0][0] != st.session_state.selected_meeting_id:
        st.session_state.selected_meeting_id = meetings[0][0]
    
    # Meeting selector
    meeting_options = [f"{m[1]} ({m[3]})" for m in meetings]
    selected_idx = st.selectbox(
        "Meeting auswählen", 
        range(len(meetings)), 
        index=current_idx,
        format_func=lambda i: meeting_options[i],
        key="meeting_selector"
    )
    
    # Update session state when selection changes
    st.session_state.selected_meeting_id = meetings[selected_idx][0]
    
    selected_meeting = meetings[selected_idx]
    meeting_id = selected_meeting[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Beiträge", selected_meeting[4])
    with col2:
        st.metric("Erkannte Stimmen", selected_meeting[5])
    with col3:
        st.metric("Status", selected_meeting[3])
    
    st.divider()
    
    # Get diarization groups
    groups = get_diarization_groups(meeting_id)
    
    if not groups:
        st.warning("Keine Beiträge für dieses Meeting gefunden.")
    else:
        # Build ONE HTML component with audio player + all seek buttons
        audio_url = get_audio_url(meeting_id)
        if audio_url:
            # Build buttons for each speaker group
            buttons_html = ""
            for group in groups:
                label = group[0]
                segments = get_segments_for_label(meeting_id, label)
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
                <audio id="meeting_audio" controls style="width: 100%;" preload="metadata">
                    <source src="{audio_url}" type="audio/mpeg">
                </audio>
                <div style="margin-top: 10px;">
                    <span style="color: #888; font-size: 12px;">Springe zu: </span>
                    {buttons_html}
                </div>
            </div>
            <script>
                var audioElement = document.getElementById('meeting_audio');
                function seekTo(seconds) {{
                    audioElement.currentTime = seconds;
                    audioElement.play();
                }}
                // Expose seekTo to parent window for cross-iframe communication
                window.parent.seekAudioTo = function(seconds) {{
                    audioElement.currentTime = seconds;
                    audioElement.play();
                }};
            </script>
            """
            components.html(audio_html, height=130)
        
        st.subheader("🎯 Sprecher zuordnen")
        st.caption(f"Schwellenwert für Auto-Matching: {SIMILARITY_THRESHOLD:.0%}")
        
        # Refresh speakers list
        speakers = get_speakers()
        speaker_options = [(None, "-- Nicht zugeordnet --")] + [(s[0], s[1]) for s in speakers]
        
        assignments_made = 0
        total_groups = len(groups)
        
        for group in groups:
            label = group[0]
            count = group[1]
            duration = group[2] or 0
            current_speaker = group[4]
            
            # Get segments for this group
            segments = get_segments_for_label(meeting_id, label) if audio_url else []
            
            # Check if already assigned
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
                    # Show clickable contribution texts
                    if audio_url and segments:
                        # Build clickable text links for each segment
                        text_links = []
                        for seg in segments:
                            seg_start = seg[2] or 0
                            seg_text = (seg[4] or "")[:100]
                            if seg_text:
                                start_fmt = f"{int(seg_start//60)}:{int(seg_start%60):02d}"
                                text_links.append(f'<a href="#" onclick="window.parent.seekAudioTo({seg_start}); return false;" style="color: #555; text-decoration: underline; text-decoration-style: dotted;" title="▶️ {start_fmt}">{seg_text}</a>')
                        
                        if text_links:
                            links_html = f"""
                            <div style="font-size: 13px; line-height: 1.6; font-style: italic; color: #666;">
                                "{' <span style="color: #ccc;">|</span> '.join(text_links)}..."
                            </div>
                            """
                            components.html(links_html, height=60 + (len(segments) // 3) * 20)
                    else:
                        sample = (group[3] or "")[:400]
                        st.caption(f"*\"{sample}...\"*" if len(sample) >= 400 else f"*\"{sample}\"*")
                    
                    # Find current speaker index
                    current_idx = 0
                    if current_speaker:
                        for i, (sid, sname) in enumerate(speaker_options):
                            if sid == current_speaker:
                                current_idx = i
                                break
                    
                    col_select, col_button, col_delete = st.columns([3, 1, 1])
                    
                    with col_select:
                        selected = st.selectbox(
                            f"Sprecher für {label}",
                            range(len(speaker_options)),
                            index=current_idx,
                            format_func=lambda i: speaker_options[i][1],
                            key=f"speaker_{label}",
                            label_visibility="collapsed"
                        )
                    
                    with col_button:
                        new_speaker_id = speaker_options[selected][0]
                        new_speaker_name = speaker_options[selected][1] if selected > 0 else ""
                        
                        if new_speaker_id and new_speaker_id != current_speaker:
                            if st.button(f"✓ Zuordnen", key=f"assign_{label}"):
                                with st.spinner(f"Verarbeite {label}... (Embedding-Extraktion kann dauern)"):
                                    # Use enhanced matching
                                    result = assign_speaker_with_matching(
                                        meeting_id, label, new_speaker_id, new_speaker_name, SIMILARITY_THRESHOLD
                                    )
                                    
                                    if result['success']:
                                        # Show detailed results
                                        msg = f"✅ Zugeordnet!"
                                        if result['embedding_extracted']:
                                            msg += f"\n🎯 Embedding extrahiert"
                                        if result.get('skipped_short', 0) > 0 and not result['embedding_extracted']:
                                            msg += f"\n⏭️ Kein Embedding (alle Segmente < 5s)"
                                        elif result.get('skipped_short', 0) > 0:
                                            msg += f"\n⏭️ {result['skipped_short']} kurze Segmente übersprungen"
                                        if result['voiceprint_saved']:
                                            msg += f"\n💾 Voiceprint gespeichert"
                                        if result['segments_tested'] > 0:
                                            msg += f"\n🔍 {result['segments_tested']} weitere Segmente getestet"
                                        if result['auto_classified'] > 0:
                                            msg += f"\n✨ {result['auto_classified']} Segmente zugeordnet"
                                        if result['kept_for_review'] > 0:
                                            msg += f"\n⚠️ {result['kept_for_review']} zur Überprüfung"
                                        if result['errors']:
                                            msg += f"\n⚠️ Hinweise: {', '.join(result['errors'][:2])}"
                                        
                                        st.success(msg)
                                        update_meeting_status(meeting_id)
                                        st.experimental_rerun()
                                    else:
                                        errors = ", ".join(result['errors']) if result['errors'] else "Unbekannter Fehler"
                                        st.error(f"Zuordnung fehlgeschlagen: {errors}")
                        elif not new_speaker_id:
                            st.button("✓ Zuordnen", key=f"assign_{label}", disabled=True)
                    
                    with col_delete:
                        if st.button("🗑️", key=f"delete_{label}", help=f"Lösche alle Beiträge von {label}"):
                            st.session_state[f'confirm_delete_{label}'] = True
                
                # Confirmation dialog outside of columns
                if st.session_state.get(f'confirm_delete_{label}', False):
                    st.warning(f"Wirklich {count} Beiträge von {label} löschen?")
                    confirm_col1, confirm_col2, confirm_col3 = st.columns([1, 1, 4])
                    with confirm_col1:
                        if st.button("✓ Ja", key=f"confirm_yes_{label}"):
                            success, message = delete_contributions(meeting_id, label)
                            if success:
                                st.success(message)
                                del st.session_state[f'confirm_delete_{label}']
                                update_meeting_status(meeting_id)
                                st.experimental_rerun()
                            else:
                                st.error(f"Fehler: {message}")
                    with confirm_col2:
                        if st.button("✗ Nein", key=f"confirm_no_{label}"):
                            del st.session_state[f'confirm_delete_{label}']
                            st.experimental_rerun()
                
                st.divider()
        
        # Progress bar
        progress = assignments_made / total_groups if total_groups > 0 else 0
        st.progress(progress, text=f"Fortschritt: {assignments_made}/{total_groups} Sprecher zugeordnet")
        
        if assignments_made == total_groups:
            st.success("✅ Alle Sprecher zugeordnet! Meeting ist bereit für den Export.")
            if st.button("🔄 Seite aktualisieren"):
                st.experimental_rerun()

# Footer
st.markdown("---")
st.caption("Meeting Transcription Agent - Speaker Classification mit Voice Matching")
