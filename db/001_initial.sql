CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS menus (
    id text PRIMARY KEY,
    restaurant_name text NOT NULL,
    currency text NOT NULL,
    source_metadata jsonb NOT NULL,
    canonical_snapshot jsonb NOT NULL,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS categories (
    menu_id text NOT NULL REFERENCES menus(id) ON DELETE CASCADE,
    id text NOT NULL,
    name text NOT NULL,
    description text,
    aliases text[] NOT NULL DEFAULT '{}',
    source_type text NOT NULL,
    source_id text NOT NULL,
    external_record_id text,
    PRIMARY KEY (menu_id, id)
);

CREATE TABLE IF NOT EXISTS menu_items (
    menu_id text NOT NULL REFERENCES menus(id) ON DELETE CASCADE,
    id text NOT NULL,
    name text NOT NULL,
    description text,
    base_price numeric,
    currency text NOT NULL,
    ingredients text[],
    allergens text[] NOT NULL DEFAULT '{}',
    dietary_attributes text[] NOT NULL DEFAULT '{}',
    availability_status text NOT NULL,
    availability_checked_at timestamptz,
    source_type text NOT NULL,
    source_id text NOT NULL,
    external_record_id text,
    embedding_text text NOT NULL,
    inferred_description jsonb,
    PRIMARY KEY (menu_id, id)
);

CREATE INDEX IF NOT EXISTS menu_items_filter_idx
    ON menu_items (menu_id, availability_status, currency, base_price);

CREATE TABLE IF NOT EXISTS item_categories (
    menu_id text NOT NULL,
    item_id text NOT NULL,
    category_id text NOT NULL,
    PRIMARY KEY (menu_id, item_id, category_id),
    FOREIGN KEY (menu_id, item_id) REFERENCES menu_items(menu_id, id) ON DELETE CASCADE,
    FOREIGN KEY (menu_id, category_id) REFERENCES categories(menu_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS item_categories_category_idx
    ON item_categories (menu_id, category_id, item_id);

CREATE TABLE IF NOT EXISTS item_variants (
    menu_id text NOT NULL,
    item_id text NOT NULL,
    id text NOT NULL,
    name text NOT NULL,
    price numeric NOT NULL,
    availability_status text NOT NULL,
    availability_checked_at timestamptz,
    source_type text NOT NULL,
    source_id text NOT NULL,
    external_record_id text,
    PRIMARY KEY (menu_id, item_id, id),
    FOREIGN KEY (menu_id, item_id) REFERENCES menu_items(menu_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS item_variants_filter_idx
    ON item_variants (menu_id, availability_status, price);

CREATE TABLE IF NOT EXISTS variation_groups (
    menu_id text NOT NULL,
    item_id text NOT NULL,
    id text NOT NULL,
    name text NOT NULL,
    description text,
    type text NOT NULL,
    is_optional boolean NOT NULL,
    min_selection integer NOT NULL,
    max_selection integer NOT NULL,
    availability_status text NOT NULL,
    availability_checked_at timestamptz,
    source_type text NOT NULL,
    source_id text NOT NULL,
    external_record_id text,
    PRIMARY KEY (menu_id, item_id, id),
    FOREIGN KEY (menu_id, item_id) REFERENCES menu_items(menu_id, id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS variation_choices (
    menu_id text NOT NULL,
    item_id text NOT NULL,
    group_id text NOT NULL,
    id text NOT NULL,
    name text NOT NULL,
    description text,
    price_adjustment numeric NOT NULL,
    availability_status text NOT NULL,
    availability_checked_at timestamptz,
    is_preselected boolean NOT NULL,
    source_type text NOT NULL,
    source_id text NOT NULL,
    external_record_id text,
    PRIMARY KEY (menu_id, item_id, group_id, id),
    FOREIGN KEY (menu_id, item_id, group_id)
        REFERENCES variation_groups(menu_id, item_id, id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS semantic_attributes (
    menu_id text NOT NULL,
    item_id text NOT NULL,
    position integer NOT NULL,
    value text NOT NULL,
    origin text NOT NULL,
    confidence double precision,
    evidence text,
    PRIMARY KEY (menu_id, item_id, position),
    FOREIGN KEY (menu_id, item_id) REFERENCES menu_items(menu_id, id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS enrichment_cache_entries (
    menu_id text NOT NULL,
    item_id text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('description', 'tag')),
    payload jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (menu_id, item_id, kind)
);

CREATE TABLE IF NOT EXISTS embedding_documents (
    menu_id text NOT NULL,
    item_id text NOT NULL,
    variant_id text NOT NULL DEFAULT '',
    embedding_model text NOT NULL,
    embedding_dimensions integer NOT NULL CHECK (embedding_dimensions > 0),
    document_text text NOT NULL,
    document_hash text NOT NULL,
    metadata jsonb NOT NULL,
    embedding vector NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (menu_id, item_id, variant_id, embedding_model),
    FOREIGN KEY (menu_id, item_id) REFERENCES menu_items(menu_id, id) ON DELETE CASCADE,
    CHECK (vector_dims(embedding) = embedding_dimensions)
);

CREATE INDEX IF NOT EXISTS embedding_documents_lookup_idx
    ON embedding_documents (menu_id, embedding_model, embedding_dimensions);

-- Inferred hints are intentionally separate from verified dietary attributes.
CREATE TABLE IF NOT EXISTS dietary_inference_cache (
    menu_id text NOT NULL,
    item_id text NOT NULL,
    payload jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (menu_id, item_id),
    CHECK (payload->>'origin' IS NOT NULL AND payload->>'origin' = 'inferred'),
    CHECK (payload->>'vegetarian' IS NOT NULL AND payload->>'vegetarian' IN ('likely', 'unlikely', 'unknown')),
    CHECK (payload->>'vegan' IS NOT NULL AND payload->>'vegan' IN ('likely', 'unlikely', 'unknown'))
);
