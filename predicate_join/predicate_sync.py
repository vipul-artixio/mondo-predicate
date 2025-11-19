import logging
from datetime import datetime
from typing import Optional

import psycopg2
import psycopg2.extras

from psycopg2.extras import Json


logger = logging.getLogger(__name__)


_INDICATIONS_QUERY = """
WITH selected_data AS (
    SELECT
        id,
        spl_id,
        spl_set_id,
        registration_number,
        product_name,
        submission_type,
        submission_number,
        submission_date,
        strength
    FROM source.usa_drug_data
    WHERE id = %s
), label_candidates AS (
    SELECT DISTINCT
        INITCAP(TRIM(BOTH '.' FROM lbl.indications_and_usage)) AS text_value,
        LENGTH(lbl.indications_and_usage) AS text_length
    FROM source.usa_drug_label lbl
    JOIN selected_data sd ON (
        sd.registration_number IS NOT NULL
        AND lbl.registration_number IS NOT DISTINCT FROM sd.registration_number
        AND sd.spl_id IS NOT NULL 
        AND lbl.spl_id = ANY(sd.spl_id)
        AND sd.spl_set_id IS NOT NULL 
        AND lbl.spl_set_id = ANY(sd.spl_set_id)
    )
    WHERE lbl.indications_and_usage IS NOT NULL
)
SELECT text_value AS indications_and_usage
FROM label_candidates
ORDER BY text_length DESC
LIMIT 1;
"""


_UPSERT_ASSESSMENT_QUERY = """
INSERT INTO drug.drug_predicate_assessments (
    ingredient_name,
    product_name,
    country_of_origin,
    approval_date,
    end_date,
    application_type,
    classification,
    registration_number,
    registration_holder,
    manufacturer,
    importer,
    generic_name,
    reference_drug,
    dosage_form,
    strength,
    route_administration,
    indication,
    therapy_area,
    other_trade_name,
    patent_information,
    distributor,
    marketing_status,
    submission_type,
    submission_number,
    submission_date,
    json_data,
    created_at,
    updated_at,
    created_by
)
VALUES (
    %(ingredient_name)s,
    %(product_name)s,
    %(country_of_origin)s,
    %(approval_date)s,
    %(end_date)s,
    %(application_type)s,
    %(classification)s,
    %(registration_number)s,
    %(registration_holder)s,
    %(manufacturer)s,
    %(importer)s,
    %(generic_name)s,
    %(reference_drug)s,
    %(dosage_form)s,
    %(strength)s,
    %(route_administration)s,
    %(indication)s,
    %(therapy_area)s,
    %(other_trade_name)s,
    %(patent_information)s,
    %(distributor)s,
    %(marketing_status)s,
    %(submission_type)s,
    %(submission_number)s,
    %(submission_date)s,
    %(json_data)s,
    %(created_at)s,
    CURRENT_TIMESTAMP,
    %(created_by)s
)
ON CONFLICT ON CONSTRAINT uq_drug_predicate_assessments_record
DO UPDATE SET
    ingredient_name = EXCLUDED.ingredient_name,
    product_name = EXCLUDED.product_name,
    country_of_origin = EXCLUDED.country_of_origin,
    approval_date = EXCLUDED.approval_date,
    end_date = EXCLUDED.end_date,
    application_type = EXCLUDED.application_type,
    classification = EXCLUDED.classification,
    registration_number = EXCLUDED.registration_number,
    registration_holder = EXCLUDED.registration_holder,
    manufacturer = EXCLUDED.manufacturer,
    importer = EXCLUDED.importer,
    generic_name = EXCLUDED.generic_name,
    reference_drug = EXCLUDED.reference_drug,
    dosage_form = EXCLUDED.dosage_form,
    strength = EXCLUDED.strength,
    route_administration = EXCLUDED.route_administration,
    indication = EXCLUDED.indication,
    therapy_area = EXCLUDED.therapy_area,
    other_trade_name = EXCLUDED.other_trade_name,
    patent_information = EXCLUDED.patent_information,
    distributor = EXCLUDED.distributor,
    marketing_status = EXCLUDED.marketing_status,
    submission_type = EXCLUDED.submission_type,
    submission_number = EXCLUDED.submission_number,
    submission_date = EXCLUDED.submission_date,
    json_data = EXCLUDED.json_data,
    updated_at = CURRENT_TIMESTAMP
RETURNING id;
"""


def _fetch_indications(cursor, data_id: int) -> Optional[str]:
    """Return aggregated indications_and_usage text for a usa_drug_data id."""

    cursor.execute(_INDICATIONS_QUERY, (data_id,))
    row = cursor.fetchone()
    if not row:
        return None
    return row.get('indications_and_usage')


def sync_predicate_assessment(cursor, data_id: int) -> Optional[int]:
    """Upsert a record into drug.drug_predicate_assessments based on usa_drug_data."""

    cursor.execute(
        """
        SELECT
            id,
            ingredient_name,
            product_name,
            country_of_origin,
            application_type,
            registration_number,
            registration_holder,
            manufacturer,
            generic_name,
            reference_drug,
            dosage_form,
            strength,
            route_administration,
            marketing_status,
            submission_type,
            submission_number,
            submission_date,
            json_data,
            created_at,
            created_by
        FROM source.usa_drug_data
        WHERE id = %s
        """,
        (data_id,),
    )
    data_row = cursor.fetchone()

    if not data_row:
        logger.warning("No source.usa_drug_data row found for id %s", data_id)
        return None

    indications = _fetch_indications(cursor, data_id)

    params = {
        'ingredient_name': data_row.get('ingredient_name'),
        'product_name': data_row.get('product_name'),
        'country_of_origin': data_row.get('country_of_origin'),
        'approval_date': None,
        'end_date': None,
        'application_type': data_row.get('application_type'),
        'classification': None,
        'registration_number': data_row.get('registration_number'),
        'registration_holder': data_row.get('registration_holder'),
        'manufacturer': data_row.get('manufacturer'),
        'importer': None,
        'generic_name': data_row.get('generic_name'),
        'reference_drug': data_row.get('reference_drug'),
        'dosage_form': data_row.get('dosage_form'),
        'strength': data_row.get('strength'),
        'route_administration': data_row.get('route_administration'),
        'indication': indications,
        'therapy_area': None,
        'other_trade_name': None,
        'patent_information': None,
        'distributor': None,
        'marketing_status': data_row.get('marketing_status'),
        'submission_type': data_row.get('submission_type'),
        'submission_number': data_row.get('submission_number'),
        'submission_date': data_row.get('submission_date'),
        'json_data': Json(data_row.get('json_data')) if data_row.get('json_data') is not None else Json({}),
        'created_at': data_row.get('created_at') or datetime.utcnow(),
        'created_by': data_row.get('created_by'),
    }

    cursor.execute(_UPSERT_ASSESSMENT_QUERY, params)
    result = cursor.fetchone()
    if not result:
        return None

    return result.get('id')


def sync_all_predicate_assessments(
    connection: psycopg2.extensions.connection,
    *,
    updated_since: Optional[datetime] = None,
    batch_size: int = 500,
) -> int:
    """
    Rebuild predicate assessments for all usa_drug_data rows (optionally filtered).

    Args:
        connection: psycopg2 connection.
        updated_since: Only process records with updated_at >= this timestamp (optional).
        batch_size: Number of usa_drug_data ids to process per transaction.

    Returns:
        Total number of usa_drug_data rows processed.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    ids_sql = """
        SELECT id
        FROM source.usa_drug_data
        {where_clause}
        ORDER BY id
    """
    where_clause = ""
    params = []
    if updated_since:
        where_clause = "WHERE updated_at >= %s"
        params.append(updated_since)

    ids_sql = ids_sql.format(where_clause=where_clause)

    id_cursor = connection.cursor()
    work_cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    total_processed = 0

    try:
        id_cursor.execute(ids_sql, params)

        while True:
            rows = id_cursor.fetchmany(batch_size)
            if not rows:
                break

            for row in rows:
                data_id = row[0] if isinstance(row, tuple) else row['id']
                sync_predicate_assessment(work_cursor, data_id)
            connection.commit()
            total_processed += len(rows)

    except Exception:
        connection.rollback()
        logger.exception("Failed during batch predicate sync; transaction rolled back.")
        raise
    finally:
        id_cursor.close()
        work_cursor.close()

    return total_processed

