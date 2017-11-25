---
layout: page
title: Archive
---

#### Blog posts

{% for post in site.categories.Work %}
  * {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ site.baseurl }}{{ post.url }})
{% endfor %}
