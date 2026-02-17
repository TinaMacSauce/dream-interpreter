@app.route("/admin/clean-sheet", methods=["POST", "OPTIONS"])
def admin_clean_sheet():
    if request.method == "OPTIONS":
        return _preflight_ok()

    if not _admin_ok(request):
        return jsonify({"error": "Unauthorized"}), 401

    if not SPREADSHEET_ID:
        return jsonify({"error": "SPREADSHEET_ID env var is not set."}), 500

    try:
        creds = _get_credentials()
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SPREADSHEET_ID)
        ws = sh.worksheet(WORKSHEET_NAME)

        values = ws.get_all_values()
        if not values or len(values) < 2:
            return jsonify({"ok": True, "message": "Sheet is empty. Nothing to clean."})

        headers_raw = values[0]
        headers_norm = [_normalize_header(h) for h in headers_raw]

        # Required columns (normalized)
        required = {"input", "spiritual meaning", "physical effects", "action", "keywords"}
        missing = [h for h in required if h not in set(headers_norm)]
        if missing:
            return jsonify({
                "error": "Missing required columns in Sheet1",
                "missing": missing,
                "found_headers": headers_raw
            }), 400

        idx = {h: headers_norm.index(h) for h in required}

        cleaned_rows = []
        changes = {
            "rows_in": len(values) - 1,
            "rows_out": 0,
            "rows_dropped_blank_input": 0,
            "cells_changed": 0,
            "keywords_normalized": 0,
            "typos_fixed": 0,
        }

        for row in values[1:]:
            # pad row to header length
            if len(row) < len(headers_norm):
                row = row + [""] * (len(headers_norm) - len(row))

            orig = row[:]  # copy

            # Clean text fields
            for col in ["input", "spiritual meaning", "physical effects", "action"]:
                i = idx[col]
                before = row[i]
                after = _fix_typos_light(before)
                if after != before:
                    changes["cells_changed"] += 1
                    # count as typo fix only if it looks like one of our known fixes occurred
                    if re.search(r"(embarass|commmun|comunity|commnity)", before or "", re.IGNORECASE):
                        changes["typos_fixed"] += 1
                row[i] = after

            # Normalize keywords
            kw_i = idx["keywords"]
            before_kw = row[kw_i]
            after_kw = _normalize_keywords_cell(before_kw)
            if after_kw != before_kw:
                changes["cells_changed"] += 1
                changes["keywords_normalized"] += 1
            row[kw_i] = after_kw

            # Drop blank input rows
            if not str(row[idx["input"]]).strip():
                changes["rows_dropped_blank_input"] += 1
                continue

            cleaned_rows.append(row)

        changes["rows_out"] = len(cleaned_rows)

        # Write back (headers + cleaned data)
        ws.clear()
        ws.update("A1", [headers_raw] + cleaned_rows)

        # Refresh cache immediately
        _load_sheet_rows(force=True)

        return jsonify({"ok": True, "message": "Sheet cleaned and updated.", "changes": changes})

    except Exception as e:
        return jsonify({"error": "Clean-sheet failed", "details": str(e)}), 500
