
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from AI_estimator import AI_estimator
from extract import get_acceleration, get_location, get_heart_rate, get_info


def main():
    pd.set_option('display.max_rows', None)
    start = datetime.now()

    acc_df = get_acceleration(fs = 10, threshold = 2)
    loc_df = get_location(T = 60, threshold = 300)
    HR_df = get_heart_rate(T = 30, threshold = 150)
    print(HR_df)
    info = get_info(HR_df)

    y = ai_estimator(acc_df, loc_df, HR_df, info, verbose = False)
    print(y)

    end = datetime.now()
    print(end - start)

ai_estimator = AI_estimator()
if __name__ == "__main__":
    main()




