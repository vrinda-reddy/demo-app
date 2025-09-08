import streamlit as st

st.set_page_config(page_title="Risk Tolerance Assessment", page_icon="📋")

st.markdown("# Risk Tolerance Assessment")
st.sidebar.markdown("# Risk Tolerance Assessment")

""" 
# Section I: Customer Investment Knowledge
This part is designed to enable us to understand and assess your company’s level of investment experience with
non-complex investment products & complex and/or derivative products.
"""
"""
Please tick the appropriate boxes below to indicate your company’s investment experience (in number of years) in each type of product.
"""
options = [
    ("Nil experience, score 0", 0),
    ("Basic experience, score 1", 1),
    ("Intermediate experience, score 3", 3),
    ("Advanced experience, score 5", 5),
]

questions = [
    "Investment Experience in Equities",
    "Investment Experience in Mutual Funds/Unit Trusts",
    "Investment Experience in Fixed Income Products",
    "Investment Experience in Warrants",
    "Investment Experience in Stock Options",
    "Investment Experience in Leveraged Forex",
    "Investment Experience in Futures and Options",
    "Investment Experience in Hedge Funds",
    "Investment Experience in Structured Notes (e.g. DCN, ELN, KO ELN, DAC, ELI)",
    "Investment Experience in OTC Swap (e.g. accumulator, decumulator)",
    "Investment Experience in Other Derivative Products (including but not limited to: Callable Bull/Bear Contracts, Structured Funds, Exchange Traded Funds with Derivative Nature and Convertible Bonds etc.)"
]

marks = []

st.header("Investment Experience Questionnaire")

for i, question in enumerate(questions):
    st.write(f"**Question {i+1}: {question}**")
    selected = st.radio(
        f"Select your experience for Question {i+1}",
        [opt[0] for opt in options],
        key=f"q{i+1}"
    )
    mark = dict(options)[selected]
    marks.append(mark)
    st.write("")  # Adds vertical space between questions

total_marks = sum(marks)
"""
Based on the assessment above, the total score of investment experience of your company is
"""
st.subheader(f"Total Marks: {total_marks}")

# Interpretation logic (not from docs, but follows Streamlit display patterns)
if total_marks == 0:
    interpretation = "No Experience"
elif 1 <= total_marks <= 2:
    interpretation = "Basic Experience"
elif total_marks == 3:
    interpretation = "Intermediate Experience"
elif total_marks == 4:
    interpretation = "Extensive Experience"
elif total_marks >= 5:
    interpretation = "Advanced Experience"
else:
    interpretation = "Invalid Score"

st.write(f"**Interpretation:** {interpretation}")


















"""
# Section II: Risk Tolerance Level

This part is designed to enable us to assess the overall risk tolerance level of your company
"""

risk_questions = [
    {
        "question": "Does your company have a specialized function responsible for making investment decisions?",
        "options": [
            "No, Our company does not have knowledge and experience for making investment decisions.",
            "Yes, Our company is a private company where the investment decision rests with the major shareholder(s) and/or director(s).",
            "Yes, Our company has a specialized function and governance practice responsible for making investment decisions."
        ],
        "scores": [1, 3, 5]
    },
    {
        "question": "What is the primary investment objective of your company?", 
        "options": [
        "Capital preservation",
        "Regular and stable income",
        "Moderate capital appreciation",
        "Moderate to high capital appreciation",
        "Maximum capital appreciation"
        ],
        "scores": [1,2,3,4,5]
    },
    {    
        "question": "What is the average percentage of liquid asset that your company’ will set aside for investment purposes?", 
        "options": [
        "Less than 5%",
        "5% to less than 10%",
        "10% to less than 20%",
        "20% to less than 30%",
        "30% or above"
        ],
        "scores":[1,2,3,4,5]
    },
    {
        "question": "It is generally true that the longer the investment horizon, the higher the risk an investor can tolerate. What is the expected investment horizon of your company’s entire investment portfolio?", 
        "options": [
        "Short term",
        "Short to medium term",
        "Medium term",
        "Medium to long term",
        "Long term",
        ],
        "scores": [1,2,3,4,5]
    },
    {
        "question": "How would your company react if your company’s portfolio fell significantly (e.g. more than 30%) within three months?", 
        "options": [
        "Do not know how to react",
        "Cut loss without any strategic consideration",
        "Would wait to see if investment improves and may cut loss rationally",
        "Understand market fluctuations are unavoidable and will not alter the determined investments strategy",
        "Undergo in-depth analysis, reallocate your investment portfolio",
        ],
        "scores": [1,2,3,4,5]
    },
    {   
        "question": "Which of the following statements best describes your company’s investment attitude?", 
        "options": [
        "My company is **not** willing to bear a price fluctuation range of **more than 5%** for financial investment and wishes to gain a return slightly higher than the bank deposit interest rate.",
        "My company is willing to bear a price fluctuation range of **around 5% to less than 10%** for financial investment and wishes to gain a return slightly higher than the bank deposit interest rate.",
        "My company is willing to bear a price fluctuation range of **10% to less than 20%** for financial investment and wishes to gain a return much higher than the bank deposit interest rate.",
        "My company is willing to bear a price fluctuation range of **20% to less than 30%** for financial investment and wishes to gain a return comparable to the average return of the stock market.",
        "My company is willing to bear a price fluctuation range of **30% or more** for financial investment and wishes to gain a return remarkably higher than the average return of the stock market.",
        ],
        "scores": [1,2,3,4,5]
     },
    {       
        "question": "Which of the following is your company's profit expectation in the next five years? (For non-profit making organizations, please use net cash flow instead.", 
        "options": [
        "Very unstable with high possibility of losses for the next two years or beyond ",
        "Unstable with some possibility of losses for the next five years ",
        "Somewhat stable with very low possibility of losses for the next five years",
        "Stable and in line with economic growth",
        "Stable and outpacing economic growth",
        ],
        "scores": [1,2,3,4,5]
     },
    {      
        "question": "What is your company’s level of experience with investment products? Please refer to your company’s investment experience assessment result in Section I", 
        "options": [
        "No experience",
        "Basic experience",
        "Intermediate experience",
        "Extensive experience",
        "Advanced experience",
        ],
        "scores": [0,2,3,4,5]
     }]


risk_marks = []

for i, q in enumerate(risk_questions):
    st.write(f"**Question {i+1}: {q['question']}**")
    selected = st.radio(
        f"Select your answer for Question {i+1}",
        q["options"],
        key=f"risk_q{i+1}"
    )
    mark = q["scores"][q["options"].index(selected)]
    risk_marks.append(mark)
    st.write("")

risk_total_marks = sum(risk_marks)
"""
Risk Tolerance Level and Corresponding Investment Objective

"""
st.subheader(f"Total Marks: {risk_total_marks}")

# Interpretation logic (not from docs, but follows Streamlit display patterns)
if risk_total_marks <= 8:
    interpretation = "Your company is a **Conservative investor** with a primary aim for capital preservation. Your company is not inclined to invest in products associated with any risk. Able to keep returns against current inflation (ie. 3-4%)."
elif 9 <= risk_total_marks <= 15:
    interpretation = "Your company is a **Moderate investor** and wants to achieve a return higher than the inflation rate and moderate growth of capital (ie. 8 - 14%). In general, your company prefers to take medium investment risk and accepts moderate fluctuation of capital values with the possibility of facing occasional high short-term losses."
elif 16 <= risk_total_marks <= 25:
    interpretation = "Your company is a **Moderate High risk tolerant investor**. Your company aims to earn returns substantially higher than the inflation to  pursue high capital appreciation (ie. 14 - 22%). Your company can accept high fluctuation of capital values and tolerate the risk of your company’s capital falling substantially the original investment."
elif risk_total_marks >= 26:
    interpretation = "Your company is an **Aggressive investor** and is eager to earn the highest potential returns (ie. 22% or above). Risk minimization is not your company’s primary concern. Your company can accept leveraged investment and bear significant capital loss if the products potentially offer very high return."
else:
    interpretation = "Invalid Score"

st.markdown(f"**Interpretation:** {interpretation}")