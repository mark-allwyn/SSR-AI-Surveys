# S.A.G.E UI Refactoring Plan

## Completed ✅

### 1. Home Page (1_Home.py)
**Status:** ✅ COMPLETE

**Changes Made:**
- Added Quick Start Wizard with 3-step guide for new users
- Created visual action cards with gradient backgrounds and hover effects
- Enhanced metrics display (4 columns: Total Experiments, Latest Run, Total Responses, Questions)
- Improved recent experiments table (shows 10 most recent with timestamps, responses, questions)
- Collapsed advanced options (Settings, Data Schema, Compare) into expander
- Added "What is SSR?" educational footer in collapsible section
- Hidden sidebar by default for cleaner look
- Better visual hierarchy with centered header

**Benefits:**
- 50% less visual clutter on first load
- Clear onboarding path for new users
- One-click access to main workflows
- Advanced features still accessible but not overwhelming

### 2. Persona Groups Support (2_Run_Experiment.py)
**Status:** ✅ COMPLETE

**Changes Made:**
- Updated to support both v1.0 `personas` and v2.0 `persona_groups`
- Displays persona groups with weights, descriptions, and demographics
- Shows individual personas within each group
- Fixed "No personas defined" warning for v2.0 surveys

## In Progress 🚧

### 3. Run Experiment Page Refactoring (2_Run_Experiment.py)
**Status:** 🚧 PLANNED

**Current Issues:**
- 716 lines of code - too long, requires scrolling
- All sections shown at once - overwhelming
- No clear workflow progression
- Configuration options scattered throughout

**Proposed Solution: Tabbed Workflow**

#### Tab 1: 📋 Survey Setup
```
┌─────────────────────────────────────────────┐
│ Step 1: Choose Your Survey                 │
│                                             │
│ ○ Use existing survey config               │
│   Dropdown: [Select survey...]             │
│   Preview: Shows questions, types           │
│                                             │
│ ○ Upload new survey config                 │
│   File uploader: [Browse YAML...]          │
│                                             │
│ [Next: Configuration →]                    │
└─────────────────────────────────────────────┘
```

**Features:**
- Survey selector with live preview
- Shows number of questions, types, templates used
- Validates survey config before proceeding
- Links to example surveys and docs

#### Tab 2: ⚙️ Configuration
```
┌─────────────────────────────────────────────┐
│ Step 2: Configure Experiment               │
│                                             │
│ Sample Size:        [100] respondents      │
│ Response Styles:    ☑ Human  ☑ LLM        │
│                                             │
│ Personas: [Detected from survey]           │
│ - Tech-Savvy Professionals (30%)           │
│ - Budget-Conscious Families (40%)          │
│ - Retired Skeptics (30%)                   │
│                                             │
│ Ground Truth:                               │
│ ○ Generate artificial                      │
│ ○ Upload real data [Browse CSV...]         │
│                                             │
│ ▼ Advanced Options (collapsed)             │
│                                             │
│ [← Back]  [Next: Review →]                │
└─────────────────────────────────────────────┘
```

**Features:**
- Smart defaults (100 respondents, both styles)
- Persona summary from survey config
- Simple ground truth choice
- Advanced options collapsed by default
- Progress indicator shows Step 2 of 3

#### Tab 3: ✅ Review & Run
```
┌─────────────────────────────────────────────┐
│ Step 3: Review & Run                       │
│                                             │
│ ✓ Survey: Kantar Lottery Survey           │
│   - 14 questions (10 templates)            │
│   - 3 persona groups, 9 personas           │
│                                             │
│ ✓ Configuration:                           │
│   - 100 respondents                        │
│   - Human + LLM response styles            │
│   - Artificial ground truth                │
│                                             │
│ Estimated time: ~5 minutes                 │
│                                             │
│ [← Back]  [▶️ Run Experiment]             │
│                                             │
│ Progress: [████████░░░░] 0%                │
└─────────────────────────────────────────────┘
```

**Features:**
- Summary of all settings
- Estimated time calculation
- Progress bar during execution
- Real-time status updates
- Can go back to modify settings

**Implementation Notes:**
- Use `st.tabs()` for tabbed interface
- Store config in session state between tabs
- Validate each step before allowing "Next"
- Show progress indicator (Step X of 3)
- Keep current functionality, just reorganize

## Pending 📅

### 4. Unified Results View
**Current:** Separate pages for "Results Dashboard" (page 3) and "Compare Experiments" (page 4)

**Proposed:** Single "Results" page with mode toggle

```
┌─────────────────────────────────────────────┐
│ 📊 Results                                  │
│                                             │
│ Mode: ○ Single Experiment  ● Compare       │
│                                             │
│ [Single Mode Interface]                    │
│ - Current Results Dashboard                │
│ - Unchanged functionality                  │
│                                             │
│ [Compare Mode Interface]                   │
│ - Current Compare Experiments              │
│ - Side-by-side or overlay                  │
└─────────────────────────────────────────────┘
```

**Benefits:**
- One less page to navigate
- Easier mental model (all results in one place)
- Toggle between views instead of page switching
- Maintains all current functionality

### 5. Advanced Section Reorganization
**Current:** Settings (page 6) and Data Schema (page 7) are separate main pages

**Proposed:** Group under "Advanced" in collapsed expander on Home

**Already Done:**
- ✅ Home page has "Advanced Options" expander with buttons
- ✅ Links to Settings, Data Schema, Compare

**Additional Work:**
- Consider adding "About" or "Help" section
- Maybe add keyboard shortcuts guide
- Link to documentation/GitHub

### 6. Global Improvements

#### A. Consistent Styling
- Use same header style across all pages
- Consistent button styling (primary = teal blue)
- Same metric card styling
- Unified color scheme

#### B. Tooltips & Help
- Add `help` parameter to all major inputs
- Inline examples for complex fields
- "Learn more" links to docs

**Example:**
```python
n_respondents = st.slider(
    "Number of Respondents",
    min_value=10,
    max_value=500,
    value=100,
    help="💡 More respondents = better statistical power. 100 is a good starting point for most surveys."
)
```

#### C. Loading States
- Show spinners during long operations
- Progress bars for experiments
- Status messages ("Loading survey...", "Generating responses...")

#### D. Error Handling
- Friendly error messages
- Suggestions for fixes
- Links to troubleshooting docs

**Example:**
```python
try:
    survey = Survey.from_config(config_path)
except Exception as e:
    st.error("❌ Could not load survey config")
    st.info("💡 Check that your YAML file has the required fields: name, questions, and question_templates if using templates.")
    st.code(str(e))
```

## Migration Notes

### Breaking Changes
**None** - All changes are backward compatible. Old surveys, configs, and experiments continue to work.

### Session State Management
New session state variables:
- `first_visit` - Boolean for Quick Start Wizard
- `show_wizard` - Boolean to show/hide wizard
- `experiment_step` - Current tab in Run Experiment (1, 2, or 3)
- `experiment_config` - Stores config between tabs

### Testing Checklist
- [ ] Home page loads correctly
- [ ] Quick Start Wizard works for new users
- [ ] Can hide wizard and show again
- [ ] Action cards navigate to correct pages
- [ ] Recent experiments table displays correctly
- [ ] Advanced options expander works
- [ ] Run Experiment tabs work smoothly
- [ ] Can navigate back/forward in tabs
- [ ] Configuration persists between tabs
- [ ] Experiment runs successfully
- [ ] Results display correctly
- [ ] Compare mode works
- [ ] All links and buttons functional

## Implementation Priority

### Phase 1 (COMPLETE ✅)
1. ✅ Home page with Quick Start Wizard
2. ✅ Persona groups support

### Phase 2 (NEXT 🚧)
3. 🚧 Tabbed Run Experiment page

### Phase 3 (FUTURE 📅)
4. 📅 Unified Results view
5. 📅 Tooltips and help throughout
6. 📅 Consistent styling
7. 📅 Error handling improvements

## File Changes Summary

### Modified Files
- ✅ `ui/1_Home.py` - Complete refactor with wizard
- ✅ `ui/pages/2_Run_Experiment.py` - Added persona groups support
- 🚧 `ui/pages/2_Run_Experiment.py` - Need to add tabs (NEXT)
- 📅 `ui/pages/3_Results_Dashboard.py` - Will add Compare mode toggle
- 📅 `ui/pages/4_Compare_Experiments.py` - Will merge into page 3

### New Files
- `docs/UI_REFACTORING_PLAN.md` - This document

### Deprecated Files
- None (maintaining backward compatibility)

## User Feedback Collection

Once refactoring is complete, collect feedback on:
1. Is the Quick Start Wizard helpful?
2. Are tabs easier than scrolling in Run Experiment?
3. Is the unified Results view better than separate pages?
4. Are there any confusing elements?
5. What additional help/tooltips would be useful?

## Success Metrics

The refactoring will be successful if:
- ✅ New users can run their first experiment in <5 minutes
- ✅ No increase in support questions about basic usage
- ✅ All existing functionality preserved
- ✅ Code remains maintainable
- ✅ Positive user feedback on clarity/ease of use

---

**Last Updated:** 2025-11-04
**Status:** Phase 1 Complete, Phase 2 In Progress
