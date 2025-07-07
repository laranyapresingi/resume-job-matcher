def get_roles():
    return ["SDE", "Data Scientist"]

def get_job_files():
    return {
        "SDE": {
            "Backend Developer": "jds_clean/sde_roles/amazon_sde.txt",
            "Frontend Developer": "jds_clean/sde_roles/chicago_sde.txt"
        },
        "Data Scientist": {
            "NLP Scientist": "jds_clean/ds_roles/cognizant_ds.txt",
            "ML Engineer": "jds_clean/ds_roles/experian_ds.txt",
        }
    }

def load_jd_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()