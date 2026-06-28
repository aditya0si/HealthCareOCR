# PLAN: Design UI to Display Results

## SECTION A — GOAL DEFINITION

The goal is to design and implement a premium, high-fidelity web dashboard interface for the Healthcare OCR and Clinical Summarization application. The UI will replace the current basic dashboard with a state-of-the-art interactive workspace showcasing transcription outputs, clinical records, document quality analysis, and real-time processing timelines.

### Observable Outcomes ("Done" Criteria):
1. **Interactive Document Health Check Panel**: Displays image preview alongside computed metrics (sharpness, contrast, brightness, noise ratio, skew angle) styled with color-coded status badges and progress meters.
2. **Premium Clinical Summary Tab**:
   - Patient Info & Clinic Card utilizing clean grids and visual headers.
   - Diagnoses rendered as interactive capsule tags.
   - Medications styled as a structured dosage table.
   - Abnormal Lab Values highlighted with color-graded warning boxes (red for HIGH, blue/amber for LOW) and status indicators.
3. **Enhanced Side-by-Side Transcription diff view**: Displays raw vs. corrected text inside code-editor style layout panels, emphasizing word-level corrections.
4. **Interactive Performance Timeline**: Visualizes execution speed and VRAM swapping flow through animated bar charts and memory usage badges.
5. **No Placeholders**: Standardizes all icons and graphics.

### Out of Scope:
- Modifying backend pipeline processing logic (OCR, scorer, summarizer models).
- Implementing multi-user login or database persistence.

---

## SECTION B — TECH STACK

- **Frontend core**: HTML5, Vanilla JavaScript (ES6+), Vanilla CSS (custom properties, flexbox/grid, glassmorphism filters, animations).
- **Backend interface**: FastAPI `/process` endpoint (in [app.py](file:///c:/Users/oliad/Desktop/HealthCareOCR/app.py)).
- **Typography & Icons**: Outfit and JetBrains Mono (Google Fonts), SVG vector icons.

---

## SECTION C — SESSION MODULARIZATION

### Session 1: Premium Stylesheet Setup (`index.css` & Global Variables)
- **Objective**: Establish the core design system and tokens in the embedded HTML styles.
- **Scope**: Stylesheet block in `app.py`.
- **Output**: Harmonious HSL colors, glassmorphism card definitions, custom scrollbars, animations.
- **Connects to**: Session 2 layout assembly.
- **Failure Surface**: Over-complicated selectors causing layout breaking on smaller screens.

### Session 2: Document Health & Quality Dashboard
- **Objective**: Render image quality metrics returned by the preprocessor.
- **Scope**: Left side-panel layout in HTML body.
- **Output**: Graphical health indicators for Sharpness, Contrast, and Skew.
- **Connects to**: Session 3 results visualization.

### Session 3: Clinical & Medication Records Tab
- **Objective**: Design interactive layouts for diagnoses, medications, and abnormal values.
- **Scope**: Clinical Summary tab in HTML body.
- **Output**: Custom cards, abnormal value highlight boxes, dosage tables.
- **Connects to**: Session 4 performance and diff panels.

### Session 4: Transcription Diff Panel & Performance Timeline
- **Objective**: Implement text diff comparison and performance stage charts.
- **Scope**: Transcription tab, Performance tab, and JS rendering code in `app.py`.
- **Output**: High-readability side-by-side text block, visual execution timeline chart.

---

## SECTION D — PROGRESS CHECKLIST

- [ ] **Session 1: Design System & Stylesheet Update**
  - [ ] Add Google Fonts (Outfit, JetBrains Mono)
  - [ ] Implement responsive layout container variables and custom scrollbars
  - [ ] Define glassmorphism utilities (`backdrop-filter`) and button hover effects
- [ ] **Session 2: Document Quality Panels**
  - [ ] Add Quality Analysis section to the left sidebar
  - [ ] Write dynamic JavaScript code to translate `quality_metrics` JSON into progress bars and status text
- [ ] **Session 3: Clinical Summarizer Layout**
  - [ ] Build Patient & Clinic Details card with modern visual cues
  - [ ] Implement Diagnoses capsule tag list
  - [ ] Design Medications prescription grid
  - [ ] Create Abnormal Lab Values table with directional icons (⬆️ / ⬇️) and warning backgrounds
- [ ] **Session 4: Text Comparison & Timing Chart**
  - [ ] Build side-by-side code blocks with scroll syncing
  - [ ] Add visual timing chart showing processing stages (Preprocessing, OCR, Swapping, LLM)
  - [ ] Verify execution by processing sample document locally and checking responsiveness
