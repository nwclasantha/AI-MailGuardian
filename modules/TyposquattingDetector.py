"""
Typosquatting / Homoglyph Detector

Detects email domains that are visually similar to legitimate email providers,
which is a common phishing and abuse technique.

Uses Levenshtein distance and homoglyph mapping to identify:
  - Misspelled domains (gmial.com, yahooo.com)
  - Character substitution (g00gle.com, micros0ft.com)
  - Unicode homoglyphs (gοοgle.com using Greek 'o')
  - TLD swaps (gmail.co instead of gmail.com)
"""

import logging
import math
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Top email providers and their domains (targets for typosquatting)
# 500+ brands across tech, finance, social media, e-commerce, government, etc.
LEGITIMATE_PROVIDERS: Dict[str, str] = {
    # === Major Email Providers ===
    'gmail.com': 'Google Gmail',
    'yahoo.com': 'Yahoo Mail',
    'outlook.com': 'Microsoft Outlook',
    'hotmail.com': 'Microsoft Hotmail',
    'live.com': 'Microsoft Live',
    'msn.com': 'Microsoft MSN',
    'aol.com': 'AOL',
    'icloud.com': 'Apple iCloud',
    'me.com': 'Apple Me',
    'mac.com': 'Apple Mac',
    'protonmail.com': 'ProtonMail',
    'proton.me': 'Proton',
    'zoho.com': 'Zoho',
    'yandex.com': 'Yandex',
    'mail.com': 'Mail.com',
    'gmx.com': 'GMX',
    'gmx.net': 'GMX',
    'fastmail.com': 'Fastmail',
    'tutanota.com': 'Tutanota',
    'tuta.io': 'Tuta',

    # === ISP Email ===
    'comcast.net': 'Comcast',
    'verizon.net': 'Verizon',
    'att.net': 'AT&T',
    'cox.net': 'Cox',
    'charter.net': 'Charter',
    'sbcglobal.net': 'SBCGlobal',
    'earthlink.net': 'EarthLink',
    'bellsouth.net': 'BellSouth',
    'spectrum.net': 'Spectrum',
    'centurylink.net': 'CenturyLink',
    'windstream.net': 'Windstream',
    'frontier.com': 'Frontier',
    'optonline.net': 'Optimum',

    # === Big Tech ===
    'microsoft.com': 'Microsoft',
    'google.com': 'Google',
    'apple.com': 'Apple',
    'amazon.com': 'Amazon',
    'meta.com': 'Meta',
    'facebook.com': 'Facebook',
    'instagram.com': 'Instagram',
    'whatsapp.com': 'WhatsApp',
    'twitter.com': 'Twitter/X',
    'x.com': 'X',
    'linkedin.com': 'LinkedIn',
    'netflix.com': 'Netflix',
    'spotify.com': 'Spotify',
    'adobe.com': 'Adobe',
    'salesforce.com': 'Salesforce',
    'oracle.com': 'Oracle',
    'sap.com': 'SAP',
    'ibm.com': 'IBM',
    'intel.com': 'Intel',
    'amd.com': 'AMD',
    'nvidia.com': 'NVIDIA',
    'cisco.com': 'Cisco',
    'vmware.com': 'VMware',
    'dell.com': 'Dell',
    'hp.com': 'HP',
    'lenovo.com': 'Lenovo',
    'samsung.com': 'Samsung',
    'sony.com': 'Sony',
    'zoom.us': 'Zoom',
    'slack.com': 'Slack',
    'dropbox.com': 'Dropbox',
    'github.com': 'GitHub',
    'gitlab.com': 'GitLab',
    'atlassian.com': 'Atlassian',
    'jira.com': 'Jira',
    'notion.so': 'Notion',
    'figma.com': 'Figma',
    'canva.com': 'Canva',
    'twilio.com': 'Twilio',
    'stripe.com': 'Stripe',
    'shopify.com': 'Shopify',
    'squarespace.com': 'Squarespace',
    'wix.com': 'Wix',
    'wordpress.com': 'WordPress',
    'godaddy.com': 'GoDaddy',
    'cloudflare.com': 'Cloudflare',
    'digitalocean.com': 'DigitalOcean',
    'heroku.com': 'Heroku',
    'mongodb.com': 'MongoDB',
    'snowflake.com': 'Snowflake',
    'palantir.com': 'Palantir',
    'servicenow.com': 'ServiceNow',
    'workday.com': 'Workday',
    'hubspot.com': 'HubSpot',
    'zendesk.com': 'Zendesk',
    'datadog.com': 'Datadog',
    'crowdstrike.com': 'CrowdStrike',
    'paloaltonetworks.com': 'Palo Alto Networks',
    'fortinet.com': 'Fortinet',
    'okta.com': 'Okta',
    'docusign.com': 'DocuSign',
    'twitch.tv': 'Twitch',
    'pinterest.com': 'Pinterest',
    'snapchat.com': 'Snapchat',
    'tiktok.com': 'TikTok',
    'reddit.com': 'Reddit',
    'discord.com': 'Discord',
    'telegram.org': 'Telegram',
    'signal.org': 'Signal',
    'skype.com': 'Skype',

    # === Social Media & Communication ===
    'tumblr.com': 'Tumblr',
    'quora.com': 'Quora',
    'medium.com': 'Medium',
    'substack.com': 'Substack',
    'flickr.com': 'Flickr',
    'vimeo.com': 'Vimeo',
    'youtube.com': 'YouTube',
    'dailymotion.com': 'Dailymotion',
    'meetup.com': 'Meetup',
    'nextdoor.com': 'Nextdoor',
    'clubhouse.com': 'Clubhouse',
    'mastodon.social': 'Mastodon',
    'threads.net': 'Threads',
    'bluesky.social': 'Bluesky',
    'wechat.com': 'WeChat',
    'line.me': 'LINE',
    'viber.com': 'Viber',
    'kik.com': 'Kik',

    # === E-commerce & Retail ===
    'ebay.com': 'eBay',
    'walmart.com': 'Walmart',
    'target.com': 'Target',
    'bestbuy.com': 'Best Buy',
    'costco.com': 'Costco',
    'homedepot.com': 'Home Depot',
    'lowes.com': 'Lowes',
    'macys.com': 'Macys',
    'nordstrom.com': 'Nordstrom',
    'kohls.com': 'Kohls',
    'wayfair.com': 'Wayfair',
    'etsy.com': 'Etsy',
    'alibaba.com': 'Alibaba',
    'aliexpress.com': 'AliExpress',
    'wish.com': 'Wish',
    'temu.com': 'Temu',
    'shein.com': 'SHEIN',
    'zappos.com': 'Zappos',
    'newegg.com': 'Newegg',
    'overstock.com': 'Overstock',
    'ikea.com': 'IKEA',
    'nike.com': 'Nike',
    'adidas.com': 'Adidas',
    'underarmour.com': 'Under Armour',
    'puma.com': 'Puma',
    'zara.com': 'Zara',
    'hm.com': 'H&M',
    'gap.com': 'Gap',
    'uniqlo.com': 'Uniqlo',
    'sephora.com': 'Sephora',
    'ulta.com': 'Ulta',

    # === Financial Services — Banks ===
    'paypal.com': 'PayPal',
    'chase.com': 'Chase Bank',
    'bankofamerica.com': 'Bank of America',
    'wellsfargo.com': 'Wells Fargo',
    'citi.com': 'Citibank',
    'citibank.com': 'Citibank',
    'usbank.com': 'US Bank',
    'pnc.com': 'PNC Bank',
    'capitalone.com': 'Capital One',
    'discover.com': 'Discover',
    'ally.com': 'Ally Bank',
    'tdbank.com': 'TD Bank',
    'td.com': 'TD',
    'hsbc.com': 'HSBC',
    'barclays.com': 'Barclays',
    'db.com': 'Deutsche Bank',
    'ubs.com': 'UBS',
    'credit-suisse.com': 'Credit Suisse',
    'goldmansachs.com': 'Goldman Sachs',
    'morganstanley.com': 'Morgan Stanley',
    'jpmorgan.com': 'JPMorgan',
    'schwab.com': 'Charles Schwab',
    'fidelity.com': 'Fidelity',
    'vanguard.com': 'Vanguard',
    'ameriprise.com': 'Ameriprise',
    'edwardjones.com': 'Edward Jones',
    'merrilledge.com': 'Merrill Edge',
    'etrade.com': 'E-Trade',
    'robinhood.com': 'Robinhood',
    'sofi.com': 'SoFi',
    'chime.com': 'Chime',
    'venmo.com': 'Venmo',
    'zelle.com': 'Zelle',
    'cashapp.com': 'Cash App',
    'wise.com': 'Wise',
    'revolut.com': 'Revolut',
    'n26.com': 'N26',
    'monzo.com': 'Monzo',
    'starlingbank.com': 'Starling Bank',

    # === Financial — Cards & Insurance ===
    'visa.com': 'Visa',
    'mastercard.com': 'Mastercard',
    'americanexpress.com': 'American Express',
    'amex.com': 'Amex',
    'statefarm.com': 'State Farm',
    'geico.com': 'GEICO',
    'progressive.com': 'Progressive',
    'allstate.com': 'Allstate',
    'libertymutual.com': 'Liberty Mutual',
    'travelers.com': 'Travelers',
    'usaa.com': 'USAA',
    'navyfederal.org': 'Navy Federal',
    'transunion.com': 'TransUnion',
    'equifax.com': 'Equifax',
    'experian.com': 'Experian',

    # === Crypto & Fintech ===
    'coinbase.com': 'Coinbase',
    'binance.com': 'Binance',
    'kraken.com': 'Kraken',
    'gemini.com': 'Gemini',
    'crypto.com': 'Crypto.com',
    'blockchain.com': 'Blockchain.com',
    'bitstamp.com': 'Bitstamp',
    'bitfinex.com': 'Bitfinex',
    'kucoin.com': 'KuCoin',
    'bybit.com': 'Bybit',
    'opensea.io': 'OpenSea',
    'metamask.io': 'MetaMask',
    'ledger.com': 'Ledger',
    'trezor.io': 'Trezor',

    # === Streaming & Entertainment ===
    'hulu.com': 'Hulu',
    'disneyplus.com': 'Disney+',
    'hbomax.com': 'HBO Max',
    'max.com': 'Max',
    'peacocktv.com': 'Peacock',
    'paramountplus.com': 'Paramount+',
    'primevideo.com': 'Prime Video',
    'crunchyroll.com': 'Crunchyroll',
    'appletv.com': 'Apple TV',
    'audible.com': 'Audible',
    'pandora.com': 'Pandora',
    'deezer.com': 'Deezer',
    'soundcloud.com': 'SoundCloud',
    'tidal.com': 'Tidal',
    'roblox.com': 'Roblox',
    'epicgames.com': 'Epic Games',
    'steampowered.com': 'Steam',
    'ea.com': 'EA',
    'playstation.com': 'PlayStation',
    'xbox.com': 'Xbox',
    'nintendo.com': 'Nintendo',
    'blizzard.com': 'Blizzard',
    'activision.com': 'Activision',
    'ubisoft.com': 'Ubisoft',

    # === Healthcare & Pharma ===
    'unitedhealth.com': 'UnitedHealth',
    'uhc.com': 'UnitedHealthcare',
    'anthem.com': 'Anthem',
    'cigna.com': 'Cigna',
    'aetna.com': 'Aetna',
    'humana.com': 'Humana',
    'kaiser.com': 'Kaiser',
    'kp.org': 'Kaiser Permanente',
    'cvs.com': 'CVS',
    'walgreens.com': 'Walgreens',
    'riteaid.com': 'Rite Aid',
    'pfizer.com': 'Pfizer',
    'modernatx.com': 'Moderna',
    'jnj.com': 'Johnson & Johnson',
    'abbvie.com': 'AbbVie',
    'merck.com': 'Merck',
    'myChart.com': 'MyChart',
    'webmd.com': 'WebMD',
    'mayoclinic.org': 'Mayo Clinic',
    'clevelandclinic.org': 'Cleveland Clinic',

    # === Government (US) ===
    'irs.gov': 'IRS',
    'ssa.gov': 'Social Security',
    'usps.com': 'USPS',
    'ups.com': 'UPS',
    'fedex.com': 'FedEx',
    'dhl.com': 'DHL',
    'uscis.gov': 'USCIS',
    'state.gov': 'US State Dept',
    'treasury.gov': 'US Treasury',
    'fbi.gov': 'FBI',
    'cia.gov': 'CIA',
    'nasa.gov': 'NASA',
    'whitehouse.gov': 'White House',
    'medicare.gov': 'Medicare',
    'va.gov': 'VA',
    'dmv.org': 'DMV',
    'login.gov': 'Login.gov',
    'usa.gov': 'USA.gov',
    'studentaid.gov': 'Student Aid',

    # === Government (International) ===
    'gov.uk': 'UK Government',
    'nhs.uk': 'NHS',
    'canada.ca': 'Canada Gov',
    'service.gov.uk': 'UK Services',
    'ato.gov.au': 'Australian Tax Office',

    # === Telecom ===
    'tmobile.com': 'T-Mobile',
    't-mobile.com': 'T-Mobile',
    'sprint.com': 'Sprint',
    'vodafone.com': 'Vodafone',
    'orange.com': 'Orange',
    'telefonica.com': 'Telefonica',
    'bt.com': 'BT',
    'ericsson.com': 'Ericsson',
    'qualcomm.com': 'Qualcomm',
    'huawei.com': 'Huawei',
    'nokia.com': 'Nokia',
    'dish.com': 'DISH',
    'xfinity.com': 'Xfinity',
    'cricket.com': 'Cricket',
    'boost.com': 'Boost Mobile',
    'mint.com': 'Mint Mobile',
    'visible.com': 'Visible',
    'ting.com': 'Ting',

    # === Airlines & Travel ===
    'delta.com': 'Delta Airlines',
    'united.com': 'United Airlines',
    'aa.com': 'American Airlines',
    'southwest.com': 'Southwest Airlines',
    'jetblue.com': 'JetBlue',
    'spirit.com': 'Spirit Airlines',
    'frontier.com': 'Frontier Airlines',
    'alaskaair.com': 'Alaska Airlines',
    'hawaiianairlines.com': 'Hawaiian Airlines',
    'britishairways.com': 'British Airways',
    'lufthansa.com': 'Lufthansa',
    'airfrance.com': 'Air France',
    'emirates.com': 'Emirates',
    'qantas.com': 'Qantas',
    'singaporeair.com': 'Singapore Airlines',
    'cathaypacific.com': 'Cathay Pacific',
    'ryanair.com': 'Ryanair',
    'easyjet.com': 'EasyJet',
    'booking.com': 'Booking.com',
    'expedia.com': 'Expedia',
    'airbnb.com': 'Airbnb',
    'vrbo.com': 'Vrbo',
    'tripadvisor.com': 'TripAdvisor',
    'kayak.com': 'Kayak',
    'priceline.com': 'Priceline',
    'hotels.com': 'Hotels.com',
    'marriott.com': 'Marriott',
    'hilton.com': 'Hilton',
    'hyatt.com': 'Hyatt',
    'ihg.com': 'IHG',
    'wyndham.com': 'Wyndham',
    'hertz.com': 'Hertz',
    'enterprise.com': 'Enterprise',
    'avis.com': 'Avis',
    'uber.com': 'Uber',
    'lyft.com': 'Lyft',

    # === Food & Delivery ===
    'doordash.com': 'DoorDash',
    'ubereats.com': 'Uber Eats',
    'grubhub.com': 'Grubhub',
    'instacart.com': 'Instacart',
    'seamless.com': 'Seamless',
    'postmates.com': 'Postmates',
    'dominos.com': 'Dominos',
    'pizzahut.com': 'Pizza Hut',
    'mcdonalds.com': 'McDonalds',
    'starbucks.com': 'Starbucks',
    'subway.com': 'Subway',
    'chipotle.com': 'Chipotle',
    'chick-fil-a.com': 'Chick-fil-A',

    # === Education ===
    'mit.edu': 'MIT',
    'harvard.edu': 'Harvard',
    'stanford.edu': 'Stanford',
    'yale.edu': 'Yale',
    'princeton.edu': 'Princeton',
    'columbia.edu': 'Columbia',
    'berkeley.edu': 'UC Berkeley',
    'ucla.edu': 'UCLA',
    'caltech.edu': 'Caltech',
    'cornell.edu': 'Cornell',
    'upenn.edu': 'UPenn',
    'nyu.edu': 'NYU',
    'ox.ac.uk': 'Oxford',
    'cam.ac.uk': 'Cambridge',
    'coursera.org': 'Coursera',
    'udemy.com': 'Udemy',
    'edx.org': 'edX',
    'khanacademy.org': 'Khan Academy',
    'duolingo.com': 'Duolingo',
    'chegg.com': 'Chegg',

    # === News & Media ===
    'nytimes.com': 'NY Times',
    'washingtonpost.com': 'Washington Post',
    'wsj.com': 'Wall Street Journal',
    'cnn.com': 'CNN',
    'bbc.com': 'BBC',
    'bbc.co.uk': 'BBC',
    'foxnews.com': 'Fox News',
    'reuters.com': 'Reuters',
    'apnews.com': 'AP News',
    'bloomberg.com': 'Bloomberg',
    'cnbc.com': 'CNBC',
    'forbes.com': 'Forbes',
    'usatoday.com': 'USA Today',
    'huffpost.com': 'HuffPost',
    'theguardian.com': 'The Guardian',
    'economist.com': 'The Economist',

    # === Cloud & Hosting ===
    'aws.amazon.com': 'AWS',
    'azure.microsoft.com': 'Azure',
    'cloud.google.com': 'Google Cloud',
    'linode.com': 'Linode',
    'vultr.com': 'Vultr',
    'namecheap.com': 'Namecheap',
    'bluehost.com': 'Bluehost',
    'hostgator.com': 'HostGator',
    'siteground.com': 'SiteGround',
    'dreamhost.com': 'DreamHost',
    'ionos.com': 'IONOS',
    'ovh.com': 'OVH',
    'hetzner.com': 'Hetzner',
    'scaleway.com': 'Scaleway',

    # === Security & VPN ===
    'norton.com': 'Norton',
    'mcafee.com': 'McAfee',
    'kaspersky.com': 'Kaspersky',
    'avast.com': 'Avast',
    'bitdefender.com': 'Bitdefender',
    'malwarebytes.com': 'Malwarebytes',
    'nordvpn.com': 'NordVPN',
    'expressvpn.com': 'ExpressVPN',
    'surfshark.com': 'Surfshark',
    'protonvpn.com': 'ProtonVPN',
    'lastpass.com': 'LastPass',
    'bitwarden.com': 'Bitwarden',
    '1password.com': '1Password',
    'dashlane.com': 'Dashlane',
    'lifelock.com': 'LifeLock',

    # === Automotive ===
    'tesla.com': 'Tesla',
    'ford.com': 'Ford',
    'gm.com': 'General Motors',
    'toyota.com': 'Toyota',
    'honda.com': 'Honda',
    'bmw.com': 'BMW',
    'mercedes-benz.com': 'Mercedes-Benz',
    'audi.com': 'Audi',
    'volkswagen.com': 'Volkswagen',
    'hyundai.com': 'Hyundai',
    'nissan.com': 'Nissan',
    'subaru.com': 'Subaru',
    'rivian.com': 'Rivian',
    'lucidmotors.com': 'Lucid Motors',
    'carvana.com': 'Carvana',
    'carmax.com': 'CarMax',

    # === Utilities & Energy ===
    'pge.com': 'PG&E',
    'sce.com': 'SoCal Edison',
    'coned.com': 'ConEd',
    'duke-energy.com': 'Duke Energy',
    'dominionenergy.com': 'Dominion Energy',
    'nexteraenergy.com': 'NextEra Energy',

    # === Real Estate ===
    'zillow.com': 'Zillow',
    'realtor.com': 'Realtor.com',
    'redfin.com': 'Redfin',
    'trulia.com': 'Trulia',
    'apartments.com': 'Apartments.com',

    # === Regional Email Providers (Europe) ===
    'web.de': 'Web.de',
    'gmx.de': 'GMX Germany',
    'freenet.de': 'Freenet',
    't-online.de': 'T-Online',
    'posteo.de': 'Posteo',
    'mailbox.org': 'Mailbox.org',
    'orange.fr': 'Orange France',
    'wanadoo.fr': 'Wanadoo',
    'laposte.net': 'La Poste',
    'free.fr': 'Free.fr',
    'sfr.fr': 'SFR',
    'virgilio.it': 'Virgilio',
    'libero.it': 'Libero',
    'alice.it': 'Alice',
    'tin.it': 'Tin.it',
    'pec.it': 'PEC Italy',
    'terra.com.br': 'Terra Brazil',
    'uol.com.br': 'UOL Brazil',
    'bol.com.br': 'BOL Brazil',
    'wp.pl': 'WP Poland',
    'onet.pl': 'Onet Poland',
    'interia.pl': 'Interia Poland',
    'seznam.cz': 'Seznam Czech',
    'centrum.cz': 'Centrum Czech',
    'mail.ru': 'Mail.ru',
    'yandex.ru': 'Yandex Russia',

    # === Regional (Asia) ===
    '163.com': 'NetEase 163',
    '126.com': 'NetEase 126',
    'sina.com': 'Sina',
    'qq.com': 'Tencent QQ',
    'sohu.com': 'Sohu',
    'aliyun.com': 'Alibaba Cloud',
    'naver.com': 'Naver',
    'daum.net': 'Daum',
    'hanmail.net': 'Hanmail',
    'yahoo.co.jp': 'Yahoo Japan',
    'rakuten.co.jp': 'Rakuten',
    'docomo.ne.jp': 'NTT Docomo',
    'rediffmail.com': 'Rediffmail',

    # === Regional (Middle East & Africa) ===
    'walla.co.il': 'Walla Israel',
    'emirates.net.ae': 'Emirates Internet',
    'safaricom.co.ke': 'Safaricom Kenya',

    # === Miscellaneous High-Value Targets ===
    'indeed.com': 'Indeed',
    'glassdoor.com': 'Glassdoor',
    'monster.com': 'Monster',
    'ziprecruiter.com': 'ZipRecruiter',
    'upwork.com': 'Upwork',
    'fiverr.com': 'Fiverr',
    'craigslist.org': 'Craigslist',
    'wikipedia.org': 'Wikipedia',
    'archive.org': 'Archive.org',
    'stackoverflow.com': 'Stack Overflow',
    'openai.com': 'OpenAI',
    'anthropic.com': 'Anthropic',
    'grammarly.com': 'Grammarly',
    'turbotax.com': 'TurboTax',
    'intuit.com': 'Intuit',
    'quickbooks.com': 'QuickBooks',
    'mint.intuit.com': 'Mint',
    'creditkarma.com': 'Credit Karma',
    'nerdwallet.com': 'NerdWallet',
    'acorns.com': 'Acorns',
    'betterment.com': 'Betterment',
    'wealthfront.com': 'Wealthfront',
}

# Known free email providers (not suspicious — do NOT flag as typosquatting)
FREE_EMAIL_PROVIDERS = {
    # Major global
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'live.com',
    'msn.com', 'aol.com', 'icloud.com', 'me.com', 'mac.com',
    'protonmail.com', 'proton.me', 'zoho.com', 'yandex.com',
    'mail.com', 'gmx.com', 'gmx.net', 'fastmail.com',
    'tutanota.com', 'tuta.io', 'yahoo.co.uk', 'yahoo.co.in',
    'yahoo.co.jp', 'googlemail.com', 'outlook.co.uk',
    'hotmail.co.uk', 'hotmail.fr', 'hotmail.de', 'hotmail.it',
    'live.co.uk', 'live.fr', 'live.de', 'live.it',
    'ymail.com', 'rocketmail.com',
    # ISP
    'comcast.net', 'verizon.net', 'att.net', 'cox.net',
    'charter.net', 'sbcglobal.net', 'earthlink.net', 'bellsouth.net',
    'optonline.net', 'frontier.com', 'windstream.net',
    'spectrum.net', 'centurylink.net',
    # Russia/CIS
    'mail.ru', 'inbox.ru', 'bk.ru', 'list.ru', 'yandex.ru',
    # Germany
    'web.de', 'gmx.de', 'freenet.de', 't-online.de',
    'posteo.de', 'mailbox.org',
    # France
    'orange.fr', 'wanadoo.fr', 'laposte.net', 'free.fr', 'sfr.fr',
    # Italy
    'virgilio.it', 'libero.it', 'alice.it', 'tin.it',
    # Brazil
    'terra.com.br', 'uol.com.br', 'bol.com.br',
    # Poland
    'wp.pl', 'onet.pl', 'interia.pl',
    # Czech
    'seznam.cz', 'centrum.cz',
    # Asia
    'rediffmail.com', '163.com', '126.com', 'sina.com', 'qq.com',
    'sohu.com', 'aliyun.com',
    'naver.com', 'daum.net', 'hanmail.net',
    'yahoo.co.jp', 'docomo.ne.jp',
    # Middle East
    'walla.co.il',
}


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _normalized_similarity(s1: str, s2: str) -> float:
    """Return similarity ratio 0.0-1.0 based on Levenshtein distance."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    distance = _levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)


class TyposquattingDetector:
    """Detects typosquatting and homoglyph attacks on email domains."""

    def __init__(self):
        self.legitimate_domains = dict(LEGITIMATE_PROVIDERS)
        self.free_providers = set(FREE_EMAIL_PROVIDERS)
        logger.info(f"TyposquattingDetector initialized with {len(self.legitimate_domains)} target domains")

    def is_free_email_provider(self, domain: str) -> bool:
        """Check if a domain is a known free email provider."""
        return domain.lower().strip() in self.free_providers

    def check_domain(self, domain: str) -> Dict:
        """
        Check a domain for typosquatting indicators.

        Returns:
            Dict with keys:
                - is_typosquat: bool - True if likely a typosquat
                - similarity: float - 0.0 to 1.0 similarity to closest legitimate domain
                - target_domain: str - the legitimate domain it most resembles
                - target_provider: str - the provider name
                - distance: int - Levenshtein edit distance
                - has_homoglyphs: bool - contains Unicode lookalike characters
                - attack_type: str - type of attack detected (if any)
        """
        domain = domain.lower().strip()
        result = {
            'is_typosquat': False,
            'similarity': 0.0,
            'target_domain': '',
            'target_provider': '',
            'distance': 999,
            'has_homoglyphs': False,
            'attack_type': 'none',
        }

        # If it IS a legitimate domain, it's not a typosquat
        if domain in self.legitimate_domains:
            result['similarity'] = 1.0
            result['target_domain'] = domain
            result['target_provider'] = self.legitimate_domains[domain]
            result['distance'] = 0
            return result

        # Check for homoglyphs first
        has_homoglyphs = self._check_homoglyphs(domain)
        result['has_homoglyphs'] = has_homoglyphs

        # Find the closest legitimate domain
        best_similarity = 0.0
        best_domain = ''
        best_provider = ''
        best_distance = 999

        domain_base = domain.rsplit('.', 1)[0] if '.' in domain else domain

        for legit_domain, provider in self.legitimate_domains.items():
            # Compare full domain
            sim = _normalized_similarity(domain, legit_domain)
            dist = _levenshtein_distance(domain, legit_domain)

            if sim > best_similarity:
                best_similarity = sim
                best_domain = legit_domain
                best_provider = provider
                best_distance = dist

            # Also compare just the base domain (before TLD)
            legit_base = legit_domain.rsplit('.', 1)[0] if '.' in legit_domain else legit_domain

            base_sim = _normalized_similarity(domain_base, legit_base)
            if base_sim > best_similarity:
                best_similarity = base_sim
                best_domain = legit_domain
                best_provider = provider
                best_distance = _levenshtein_distance(domain_base, legit_base)  # Use base-domain distance, consistent with base_sim

        result['similarity'] = best_similarity
        result['target_domain'] = best_domain
        result['target_provider'] = best_provider
        result['distance'] = best_distance

        # Determine if this is a typosquat
        if has_homoglyphs and best_similarity > 0.7:
            result['is_typosquat'] = True
            result['attack_type'] = 'homoglyph'
        elif best_distance == 1 and best_similarity >= 0.85:
            # Single character edit from a known domain - very suspicious
            result['is_typosquat'] = True
            result['attack_type'] = 'single_char_edit'
        elif best_distance == 2 and best_similarity >= 0.75:
            # Two character edits - suspicious (covers character swaps like gmial→gmail)
            result['is_typosquat'] = True
            result['attack_type'] = 'double_char_edit'
        elif best_similarity >= 0.90 and domain != best_domain:
            # Very high similarity
            result['is_typosquat'] = True
            result['attack_type'] = 'high_similarity'
        elif self._check_tld_swap(domain, best_domain):
            result['is_typosquat'] = True
            result['attack_type'] = 'tld_swap'

        return result

    def _check_homoglyphs(self, domain: str) -> bool:
        """Check if the domain contains Unicode homoglyph characters."""
        for char in domain:
            # Check if character is non-ASCII
            if ord(char) > 127:
                return True
        # Check common digit-letter substitutions in the base domain
        domain_base = domain.rsplit('.', 1)[0] if '.' in domain else domain
        # Patterns like g00gle, micros0ft, yah00
        if re.search(r'[a-z]0[a-z]', domain_base):  # letter-zero-letter
            return True
        # Digit-one substitution (e.g., m1crosoft) — only flag if a single digit present
        # to avoid false positives on legitimate domains like mail1service.com
        if re.search(r'(?<=[a-z])1(?=[a-z])', domain_base):
            if len(re.findall(r'\d', domain_base)) == 1:
                return True
        return False

    def _check_tld_swap(self, domain: str, target: str) -> bool:
        """Check if this is the same base domain with a different TLD."""
        if '.' not in domain or '.' not in target:
            return False

        domain_parts = domain.rsplit('.', 1)
        target_parts = target.rsplit('.', 1)

        # Same base, different TLD
        if domain_parts[0] == target_parts[0] and domain_parts[1] != target_parts[1]:
            return True

        return False

    def get_typosquat_score(self, domain: str) -> float:
        """
        Return a typosquatting risk score 0.0 (safe) to 1.0 (definitely typosquat).

        This is the main feature for ML integration.
        """
        result = self.check_domain(domain)

        if not result['is_typosquat']:
            # Exact match to a legitimate domain — zero risk
            if result['distance'] == 0:
                return 0.0
            # Even if not flagged, return the raw similarity as a signal
            # But dampen it - only high similarities should contribute
            if result['similarity'] > 0.7 and result['distance'] <= 3:
                return result['similarity'] * 0.5  # Partial signal
            return 0.0

        # It IS a typosquat - score by attack type
        if result['attack_type'] == 'homoglyph':
            return 0.95
        elif result['attack_type'] == 'single_char_edit':
            return 0.90
        elif result['attack_type'] == 'tld_swap':
            return 0.85
        elif result['attack_type'] == 'high_similarity':
            return 0.80
        elif result['attack_type'] == 'double_char_edit':
            return 0.75

        return result['similarity']

    def get_shannon_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of a string.

        High entropy suggests random/automated generation.
        Typical human-chosen local parts: 2.5-4.0
        Random generated strings: 4.0-5.0+
        """
        if not text:
            return 0.0

        length = len(text)
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1

        entropy = 0.0
        for count in freq.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy
