from . import education, income, marital, occupation, relationship, workclass

TARGETS = [income.TARGET, education.TARGET, occupation.TARGET, relationship.TARGET, workclass.TARGET, marital.TARGET]
BY_SLUG = {t.slug: t for t in TARGETS}
