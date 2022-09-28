from django import forms

class SearchForm(forms.Form):
    path = forms.CharField(label='Path', max_length=100)
    query = forms.CharField(label='Query', max_length=100)