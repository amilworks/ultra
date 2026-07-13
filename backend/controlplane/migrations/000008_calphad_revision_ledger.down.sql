DO $$
BEGIN
  RAISE EXCEPTION 'migration 000008 is irreversible: CALPHAD audit ledger tables must not be dropped'
    USING ERRCODE = '0A000';
END;
$$;
