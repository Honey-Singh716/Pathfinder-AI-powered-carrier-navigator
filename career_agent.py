import json
from typing import Dict, List

from rich.console import Console
from rich.table import Table
from career_model import predict as model_predict, ROADMAPS


console = Console()


# Model logic now lives in career_model.py


def prompt_user() -> Dict:
    console.print("Enter your details (press Enter to accept defaults in brackets):", style="bold cyan")

    def ask_float(msg: str, default: float) -> float:
        try:
            val = input(f"{msg} [{default}]: ").strip()
            return float(val) if val else default
        except ValueError:
            console.print("Invalid number, using default.", style="yellow")
            return default

    def ask_choice(msg: str, choices: List[str], default: str) -> str:
        console.print(f"Choices: {', '.join(choices)}")
        val = input(f"{msg} [{default}]: ").strip()
        return val if val in choices else default

    def ask_bool(msg: str, default: bool) -> int:
        val = input(f"{msg} (y/n) [{'y' if default else 'n'}]: ").strip().lower()
        if val in ("y", "yes"): return 1
        if val in ("n", "no"): return 0
        return 1 if default else 0

    interests = [
        "Data Science",
        "Software Development",
        "Web Development",
        "Machine Learning",
        "Design",
        "Electronics",
        "Cybersecurity",
        "Cloud",
        "Business",
    ]

    cgpa = ask_float("CGPA (0-10)", 8.0)
    interest = ask_choice("Primary interest", interests, "Software Development")
    subject = input("Subject (optional) []: ").strip()
    skill = ask_float("General Skill level (0-10)", 7.0)
    logical_skill = ask_float("Logical Skill (0-10)", 7.0)
    creativity = ask_float("Creativity (0-10)", 7.0)
    communication = ask_float("Communication (0-10)", 7.0)

    console.print("Skills (y/n):", style="bold")
    skills_python = ask_bool("Python", True)
    skills_ml = ask_bool("Machine Learning", False)
    skills_web = ask_bool("Web Dev", False)
    skills_ds = ask_bool("Data Science", False)

    console.print("Subjects you like (y/n):", style="bold")
    subject_math = ask_bool("Mathematics", True)
    subject_cs = ask_bool("Computer Science", True)
    subject_electronics = ask_bool("Electronics", False)
    subject_design = ask_bool("Design/Art", False)

    return {
        "cgpa": cgpa,
        "interest": interest,
        "subject": subject,
        "skill": skill,
        "logical_skill": logical_skill,
        "creativity": creativity,
        "communication": communication,
        "skills_python": skills_python,
        "skills_ml": skills_ml,
        "skills_web": skills_web,
        "skills_ds": skills_ds,
        "subject_math": subject_math,
        "subject_cs": subject_cs,
        "subject_electronics": subject_electronics,
        "subject_design": subject_design,
    }


def predict_and_show(user_row: Dict):
    result = model_predict(user_row)

    console.print("\nPrediction:", style="bold green")
    console.print(f"Recommended career: [bold]{result['career']}[/bold]")

    if result.get("top"):
        table = Table(title="Top matches")
        table.add_column("Career")
        table.add_column("Probability")
        for c, p in result["top"]:
            console.print("")
            table.add_row(c, f"{p:.2f}")
        console.print(table)

    roadmap = result.get("roadmap", ROADMAPS.get(result['career'], ["Explore foundational skills and build 2-3 projects."]))
    console.print("\nRoadmap:", style="bold blue")
    for i, step in enumerate(roadmap, 1):
        console.print(f"{i}. {step}")


def main():
    user_row = prompt_user()
    predict_and_show(user_row)


if __name__ == "__main__":
    main()
