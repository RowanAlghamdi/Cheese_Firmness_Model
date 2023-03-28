def feature_plot(data, grouping_col, feature, bar_plot=None):
    # Importing the required libraries needed for this function
    import altair as alt
    import pandas as pd

    """
   Given a dataframe, a grouping column, the column of a numeric feature and the option of plotting, return a summary
   of the results, dataframe that has been grouped by a chosen column with the mean function applied to the feature. Returning a
   bar plot is optional.
    
    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        The dataframe 
    grouping_col : Union[int, str]
        The column to group the data on
    feature : Union[int, float]
        After grouping, the feature column to applying the mean fucntion to
    bar_plot : 'show', optional
        The bar plot, with the option 'show', of grouping_col and the values of feature. The default value is None.
        
    Returns
    -------
    str
        A summery of the results computed
    df_grouped : pandas.core.frame.DataFrame 
        A dataframe with the group by column and the mean function applied to chosen column
    altair.vegalite.v4.api.Chart
        A bar plot of grouping_col and the values of the feature on the y-axis
        
    Raises
    ------
    TypeError
        If the input argument data is not of type pandas.core.frame.DataFrame
    AssertError
        If the input argument grouping_col is not part of the data columns
    AssertError
        If the input argument feature is not part of the data columns
    
    Examples
    --------
    >>> feature_plot(Customers ,'Quebec', 'Net_revenue', bar_plot='show')
    'Montreal in Quebec have the highest avergae Net_revenue of 136700.7'
    'Laval in Quebec have the lowest avergae Net_revenue of 3980.0'
    
    {'Quebec': ['Montreal', 'Quebec City', 'Sherbrooke', 'Levis', 'Laval'], 
        'net_revenue': [136700.7,49088.0, 99000.0, 100473.4, 3980.0]}
        
    Alt.Chart(...)
    """
    # Checks if a dataframe is the type of object being passed into the data argument
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The data argument is not of type DataFrame")
    # Checks that the grouping_col is in the dataframe
    if not grouping_col in data.columns:
        raise Exception(
            "The"
            + str(grouping_col)
            + "does not exist in the dataframe. Check column names again."
        )
    # Checks that the feature is in the dataframe
    if not feature in data.columns:
        raise Exception(
            "The"
            + str(feature)
            + "does not exist in the dataframe. Check column names again."
        )

    # Use groupby function on the desired column and applying the mean function to feature
    df_grouped = (
        (data.groupby(grouping_col)[[feature]].mean())
        .reset_index()
        .sort_values(by=feature, ascending=False)
    )

    # Print a summary of the results found when applying the function
    print(
        df_grouped.iloc[0, 0],
        "in",
        grouping_col,
        "have the highest avergae",
        feature,
        "of",
        df_grouped.iloc[0, 1],
    )
    print(
        df_grouped.iloc[-1, 0],
        "in",
        grouping_col,
        "have the lowest avergae",
        feature,
        "of",
        df_grouped.iloc[-1, 1],
    )

    if bar_plot == "show":

        # Use altair to generate a bar plot
        plot1 = (
            alt.Chart(df_grouped, width=500, height=300)
            .mark_bar(color="pink")
            .encode(
                x=alt.X(grouping_col, sort="y", title=str(grouping_col).capitalize()),
                y=alt.Y(feature, title=str(feature).capitalize()),
            )
            .properties(
                title=(
                    str(grouping_col).capitalize() + " vs. " + str(feature).capitalize()
                )
            )
        )

        plot1.display()

    elif bar_plot is None:
        pass

    return df_grouped
